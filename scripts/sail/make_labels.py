"""SAIL step 1/3 — build reached-pose actions + the AWE precision channel.

For each demo episode:
  1. reached action[t] = (EE pos[t+1], EE yaw[t+1], commanded gripper[t])
     — the paper's 4.2 controller-invariant target, mirroring FastIL's
     add_reached_actions.py. The gripper stays *commanded*, as FastIL does.
  2. AWE waypoints via dp_waypoint_selection (err_threshold=0.005, FastIL
     default), with the gripper channel binarised so open/close events are
     forced to be waypoints.
  3. per-frame precision via DBSCAN(eps=0.025, min_samples=3) over waypoint
     positions plus gap-fill — FastIL label_awe_trajectory_precision.py
     CLUSTERING mode.

Writes {beyond_demo}/{task}/sail/awe/<name>.h5 holding reached_action (N,5),
precision (N,), action (N,6 — the training key), ep_offset/ep_len, and
ExternalLinks to the source demo h5 for pixels/wrist_pixels, so the file stays
~12 MB instead of copying 77 GB of frames. It is a drop-in dataset_path for
CubeAWImageDataset once shape_meta.action.shape is [6].

Only cube is wired up: reached-pose extraction needs task-specific proprio
keys, and pusht/slippery store a 2-D agent position rather than an EE pose.

Usage:
  .venv/bin/python scripts/sail/make_labels.py [--n-eps 800] [--workers 16]
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.environ.get("AWE_ROOT", str(Path(__file__).resolve().parents[2] / "third_party" / "awe")))

# AWE's traj_reconstruction only needs robosuite for quat_slerp — shim it so we
# don't need robosuite in .venv (scipy Slerp is numerically equivalent).
def _quat_slerp(quat_start, quat_end, fraction):
    from scipy.spatial.transform import Rotation as _R, Slerp as _Slerp
    rots = _R.from_quat(np.stack([quat_start, quat_end]))
    return _Slerp([0.0, 1.0], rots)([float(fraction)]).as_quat()[0]

_rs = types.ModuleType("robosuite")
_rs_utils = types.ModuleType("robosuite.utils")
_rs_tu = types.ModuleType("robosuite.utils.transform_utils")
_rs_tu.quat_slerp = _quat_slerp
_rs.utils = _rs_utils
_rs_utils.transform_utils = _rs_tu
sys.modules.setdefault("robosuite", _rs)
sys.modules.setdefault("robosuite.utils", _rs_utils)
sys.modules.setdefault("robosuite.utils.transform_utils", _rs_tu)

from waypoint_extraction.extract_waypoints import dp_waypoint_selection

_HOME = os.environ["STABLEWM_HOME"]
SRC = os.environ.get("SAIL_SRC", f"{_HOME}/cube/data/cube_multiview_diffusion_c03.h5")
OUT = os.environ.get("SAIL_OUT", f"{_HOME}/cube/sail/awe/cube_reached_precision_c03.h5")
ERR_THRESHOLD = 0.005   # FastIL save_awe_waypoint default
DBSCAN_EPS = 0.025      # FastIL label_awe_trajectory_precision
DBSCAN_MIN_SAMPLES = 3


def process_episode(args):
    ep, i0, ln = args
    with h5py.File(SRC, "r") as f:
        pos = f["proprio_effector_pos"][i0:i0 + ln]          # (ln, 3)
        yaw = f["proprio_effector_yaw"][i0:i0 + ln, 0]       # (ln,)
        acts = np.nan_to_num(f["action"][i0:i0 + ln].astype(np.float64), nan=0.0)

    # --- 1. reached-pose actions: target = pose at t+1, gripper stays commanded ---
    nxt = np.minimum(np.arange(ln) + 1, ln - 1)
    reached = np.stack(
        [pos[nxt, 0], pos[nxt, 1], pos[nxt, 2], yaw[nxt], acts[:, 4]], axis=1
    ).astype(np.float32)                                      # (ln, 5)

    # --- 2. AWE waypoints (gripper binarized so events force waypoints) ---
    grip_bin = np.where(acts[:, 4] >= 0, 1.0, -1.0)
    awe_actions = np.concatenate([pos, grip_bin[:, None]], axis=1)  # (ln, 4)
    quat = Rotation.from_euler("z", yaw).as_quat()                  # (ln, 4) xyzw
    gt_states = [
        {"robot0_eef_pos": pos[i], "robot0_eef_quat": quat[i]} for i in range(ln)
    ]
    waypoints = dp_waypoint_selection(
        actions=awe_actions, gt_states=gt_states, err_threshold=ERR_THRESHOLD
    )
    waypoints = np.asarray(sorted(set(int(w) for w in waypoints)), dtype=np.int64)

    # --- 3. precision via DBSCAN over waypoint positions + gap fill ---
    wp_pos = awe_actions[waypoints][:, :3]
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(wp_pos).labels_
    precision = np.zeros(ln, dtype=np.float32)
    precision[waypoints] = (labels >= 0).astype(np.float32)
    for j in range(len(labels) - 1):
        if labels[j] >= 0 and labels[j + 1] >= 0:
            precision[waypoints[j]:waypoints[j + 1] + 1] = 1.0

    return ep, reached, precision, len(waypoints)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=str, default=OUT)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"[sail-labels/cube] src={SRC} -> {args.out}", flush=True)

    with h5py.File(SRC, "r") as f:
        ep_offset = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
    n_eps = args.n_eps or len(ep_len)
    jobs = [(ep, int(ep_offset[ep]), int(ep_len[ep])) for ep in range(n_eps)]

    n_total = int(ep_offset[n_eps - 1] + ep_len[n_eps - 1])
    reached_all = np.zeros((n_total, 5), dtype=np.float32)
    precision_all = np.zeros(n_total, dtype=np.float32)
    n_wp = []

    with Pool(args.workers) as pool:
        for ep, reached, precision, nw in pool.imap_unordered(process_episode, jobs):
            i0, ln = int(ep_offset[ep]), int(ep_len[ep])
            reached_all[i0:i0 + ln] = reached
            precision_all[i0:i0 + ln] = precision
            n_wp.append(nw)
            if len(n_wp) % 50 == 0:
                print(f"{len(n_wp)}/{n_eps} eps  waypoints/ep={np.mean(n_wp):.1f}  "
                      f"precision_frac={precision_all[:i0+ln].mean():.3f}", flush=True)

    action6 = np.concatenate([reached_all, precision_all[:, None]], axis=1)

    with h5py.File(args.out, "w") as g:
        g["reached_action"] = reached_all
        g["precision"] = precision_all
        g["action"] = action6            # 6-D training key (drop-in for adapter)
        g["ep_offset"] = ep_offset[:n_eps]
        g["ep_len"] = ep_len[:n_eps]
        g["pixels"] = h5py.ExternalLink(SRC, "/pixels")
        g["wrist_pixels"] = h5py.ExternalLink(SRC, "/wrist_pixels")
        g.attrs["source"] = SRC
        g.attrs["err_threshold"] = ERR_THRESHOLD
        g.attrs["dbscan_eps"] = DBSCAN_EPS
        g.attrs["dbscan_min_samples"] = DBSCAN_MIN_SAMPLES

    print(f"\nDONE. eps={n_eps}  waypoints/ep={np.mean(n_wp):.1f}±{np.std(n_wp):.1f}  "
          f"precision_frac={precision_all.mean():.3f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
