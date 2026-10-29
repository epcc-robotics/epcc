"""SAIL step 1/3 for the PushT family — reached-pose actions + AWE precision.

MIRROR of ``scripts/sail/make_labels.py`` (cube), which states up front that
"Only cube is wired up: reached-pose extraction needs task-specific proprio
keys, and pusht/slippery store a 2-D agent position rather than an EE pose."
That file is left untouched; the cube assumptions run too deep to branch
cleanly — it reads ``proprio_effector_pos``/``_yaw``, builds quaternions for the
rotational AWE channel, carries a commanded gripper column, and links a wrist
camera. PushT has none of those.

What survives the port, and what changes:

    reached action   cube [x,y,z,yaw,grip] (5)      pusht [x,y] (2)
    source           proprio_effector_pos/_yaw      pos_agent  (already in the h5)
    AWE mode         geometric (pos + quat)         pos_only=True
    gripper events   force waypoints                none exist
    DBSCAN space     metres                         pixels (see SCALE below)
    views            pixels + wrist_pixels          pixels only
    training key     action (6) = 5 + precision     action (3) = 2 + precision

🔑 The reached target is ``pos_agent[t+1]``, matching the cube file's
``pos[nxt]`` and the off-by-one that cost the SkiP port a debugging round: at
inference the executed action is the FIRST element of the chunk while the env
sits at frame t, so it must name the NEXT pose.

🔑 The two AWE constants are lengths, so they cannot carry over unchanged from a
metres-based task to a pixels-based one. FastIL's defaults (err 0.005,
DBSCAN eps 0.025) sit at roughly 1% and 5% of the cube effector's working
extent; the same fractions of PushT's 512 px arena give 5.12 and 25.6. Both are
flags so the choice can be swept rather than trusted — ``--err-threshold`` and
``--dbscan-eps``.

⚠️ 3-D, not 2-D, even when the AWE scheduler is off: ReachedPoseSAILPolicy
checks ``action.shape[-1] == env_action_dim + 1`` unconditionally
(``policy/sail.py:239``), so the precision column must be present regardless of
``use_precision``.

Usage:
  python scripts/sail/make_labels_pusht.py --task slippery [--n-eps 200] [--workers 16]
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
from sklearn.cluster import DBSCAN

AWE_ROOT = os.environ.get("AWE_ROOT", str(Path(__file__).resolve().parents[2] / "third_party" / "awe"))
sys.path.insert(0, AWE_ROOT)

# Same robosuite shim as the cube file: AWE's traj_reconstruction imports
# robosuite only for quat_slerp, which the pos_only path never calls — but the
# import happens at module load, so it still has to resolve. scipy Slerp is
# numerically equivalent.
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

from waypoint_extraction.extract_waypoints import dp_waypoint_selection  # noqa: E402

HOME = os.environ["STABLEWM_HOME"]

# PARK demo sets — the generation DP / DTG-MPCC / SkiP are all trained on.
TASKS = {
    "plain": dict(
        home="pusht",
        src=f"{HOME}/datasets/expert_slow_025_n200_centered_v2_park.h5",
        out="pusht_reached_precision_park.h5",
    ),
    "slippery": dict(
        home="slippery_pusht",
        src=f"{HOME}/slippery_pusht/data/expert_slow_050_n200_d035_park.h5",
        out="slippery_reached_precision_park.h5",
    ),
}

ARENA_PX = 512.0
# FastIL metres-based defaults rescaled by working extent (see module docstring).
ERR_THRESHOLD = 0.01 * ARENA_PX      # 5.12 px
DBSCAN_EPS = 0.05 * ARENA_PX         # 25.6 px
DBSCAN_MIN_SAMPLES = 3

_CFG: dict = {}


def _init(cfg):
    global _CFG
    _CFG = cfg


def process_episode(args):
    ep, i0, ln = args
    with h5py.File(_CFG["src"], "r") as f:
        pos = np.nan_to_num(f["pos_agent"][i0:i0 + ln].astype(np.float64), nan=0.0)  # (ln,2)

    # --- 1. reached-pose actions: target = agent position at t+1 ---
    nxt = np.minimum(np.arange(ln) + 1, ln - 1)
    reached = pos[nxt].astype(np.float32)                      # (ln, 2)

    # --- 2. AWE waypoints. pos_only=True: no gripper column to binarise, and
    #        point_line_distance is dimension-agnostic (dot/norm), so the 2-D
    #        positions go in directly as both actions and gt_states.
    waypoints = dp_waypoint_selection(
        actions=pos, gt_states=pos,
        err_threshold=_CFG["err_threshold"], pos_only=True,
    )
    waypoints = np.asarray(sorted(set(int(w) for w in waypoints)), dtype=np.int64)

    # --- 3. precision via DBSCAN over waypoint positions + gap fill ---
    # Dense clusters of waypoints = the trajectory needed many keyframes there =
    # a precision phase. Identical to the cube file, only the space is pixels.
    labels = DBSCAN(eps=_CFG["dbscan_eps"],
                    min_samples=DBSCAN_MIN_SAMPLES).fit(pos[waypoints]).labels_
    precision = np.zeros(ln, dtype=np.float32)
    precision[waypoints] = (labels >= 0).astype(np.float32)
    for j in range(len(labels) - 1):
        if labels[j] >= 0 and labels[j + 1] >= 0:
            precision[waypoints[j]:waypoints[j + 1] + 1] = 1.0

    return ep, reached, precision, len(waypoints)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--n-eps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--err-threshold", type=float, default=ERR_THRESHOLD)
    ap.add_argument("--dbscan-eps", type=float, default=DBSCAN_EPS)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    T = TASKS[a.task]
    src = T["src"]
    if not Path(src).exists():
        raise SystemExit(f"demo h5 missing: {src}")
    out_dir = Path(HOME) / T["home"] / "sail" / "awe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else out_dir / T["out"]

    cfg = dict(src=src, err_threshold=a.err_threshold, dbscan_eps=a.dbscan_eps)
    print(f"[sail-labels/{a.task}] src={src}\n  err_threshold={a.err_threshold:.3f} px  "
          f"dbscan_eps={a.dbscan_eps:.3f} px  -> {out}", flush=True)

    with h5py.File(src, "r") as f:
        ep_offset = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
    n_eps = a.n_eps or len(ep_len)
    jobs = [(ep, int(ep_offset[ep]), int(ep_len[ep])) for ep in range(n_eps)]

    n_total = int(ep_offset[n_eps - 1] + ep_len[n_eps - 1])
    reached_all = np.zeros((n_total, 2), dtype=np.float32)
    precision_all = np.zeros(n_total, dtype=np.float32)
    n_wp = []

    with Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        for ep, reached, precision, nw in pool.imap_unordered(process_episode, jobs):
            i0, ln = int(ep_offset[ep]), int(ep_len[ep])
            reached_all[i0:i0 + ln] = reached
            precision_all[i0:i0 + ln] = precision
            n_wp.append(nw)
            if len(n_wp) % 25 == 0:
                print(f"  {len(n_wp)}/{n_eps} eps  waypoints/ep={np.mean(n_wp):.1f}  "
                      f"precision_frac={precision_all[:i0+ln].mean():.3f}", flush=True)

    action3 = np.concatenate([reached_all, precision_all[:, None]], axis=1)

    with h5py.File(out, "w") as g:
        g["reached_action"] = reached_all
        g["precision"] = precision_all
        g["action"] = action3            # 3-D training key: [x, y, precision]
        g["ep_offset"] = ep_offset[:n_eps]
        g["ep_len"] = ep_len[:n_eps]
        # External link keeps this file ~MBs instead of copying the frames.
        g["pixels"] = h5py.ExternalLink(src, "/pixels")
        g["pos_agent"] = h5py.ExternalLink(src, "/pos_agent")
        g["state"] = h5py.ExternalLink(src, "/state")
        g.attrs["source"] = src
        g.attrs["task"] = a.task
        g.attrs["err_threshold"] = a.err_threshold
        g.attrs["dbscan_eps"] = a.dbscan_eps
        g.attrs["dbscan_min_samples"] = DBSCAN_MIN_SAMPLES

    print(f"\nDONE {a.task}. eps={n_eps}  waypoints/ep={np.mean(n_wp):.1f}±{np.std(n_wp):.1f}  "
          f"precision_frac={precision_all.mean():.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
