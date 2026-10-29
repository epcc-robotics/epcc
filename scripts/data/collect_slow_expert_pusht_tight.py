"""Collect slow PushT demos with a TIGHT success criterion (pos<5, ang<π/36).

Fixes from the original collect_slow_expert_pusht.py:
  1. Tight collection threshold: pos<5 px AND ang<π/36 rad (vs canonical pos<20, ang<π/9).
     Demos park the block at the center of the goal, not at the boundary.
  2. Failed attempts (truncated by max_steps) are filtered out post-collection.
     Only `terminated=True` (tight-success) episodes survive into the final dataset.
  3. NaN-at-last-action bug: drop last frame of every episode in post-processing.
  4. Reports avg velocity (px/step) at end. Target for slow=0.25: ~6 px/step.

Strategy:
  - Step 1: record_dataset(episodes=N_REQUEST) into a RAW temp file with tight `eval_state`.
    Both terminated (=tight success) and truncated (=max_steps exceeded) attempts written.
  - Step 2: post-process — filter to only `terminated` last-frame, drop NaN last frame,
    take first ``num_traj`` survivors, write to FINAL dataset.

Usage:
    python scripts/data/collect_slow_expert_pusht_tight.py \
        slowdown_factor=0.25 num_traj=200 \
        dataset_name=expert_dataset_025_n200_centered \
        world.max_episode_steps=500 n_request=800
"""
import os, sys, types
os.environ.setdefault("MUJOCO_GL", "egl")

# lerobot groot monkey-patch
groot_mod = types.ModuleType("lerobot.policies.groot")
groot_mod.configuration_groot = types.ModuleType("lerobot.policies.groot.configuration_groot")
groot_mod.configuration_groot.GrootConfig = type("GrootConfig", (), {})
sys.modules["lerobot.policies.groot"] = groot_mod
sys.modules["lerobot.policies.groot.configuration_groot"] = groot_mod.configuration_groot
sys.modules["lerobot.policies.groot.__init__"] = groot_mod
sys.modules["lerobot.policies.groot.modeling_groot"] = types.ModuleType("lerobot.policies.groot.modeling_groot")
sys.modules["lerobot.policies.groot.modeling_groot"].GrootPolicy = type("GrootPolicy", (), {})

from pathlib import Path
_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import hdf5plugin  # noqa: F401  — must come before h5py
import h5py
import hydra
import numpy as np
import stable_worldmodel as swm
from omegaconf import DictConfig
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from collect_slow_expert_pusht import LeRobotDPPolicy


def patch_pusht_tight_success(world, pos_thresh: float = 5.0, ang_thresh: float = np.pi / 36):
    """Override env eval_state to use TIGHT block-only criterion."""
    sync = world.envs.unwrapped
    for env in sync.envs:
        base = env
        while hasattr(base, "env"):
            base = base.env

        def _tight_eval_state(goal_state, cur_state, _pt=pos_thresh, _at=ang_thresh):
            block_pos_diff = np.linalg.norm(goal_state[2:4] - cur_state[2:4])
            angle_diff_raw = np.abs(goal_state[4] - cur_state[4])
            angle_diff = min(angle_diff_raw, 2 * np.pi - angle_diff_raw)
            success = block_pos_diff < _pt and angle_diff < _at
            state_dist = np.linalg.norm(goal_state - cur_state)
            return success, state_dist

        base.eval_state = _tight_eval_state
        _orig_reset = base.reset
        def _patched_reset(*args, _base=base, _orig=_orig_reset, **kwargs):
            result = _orig(*args, **kwargs)
            _base.goal_state[2] = _base.goal_pose[0]
            _base.goal_state[3] = _base.goal_pose[1]
            _base.goal_state[4] = _base.goal_pose[2]
            return result
        base.reset = _patched_reset


def _filter_and_finalize(raw_path: Path, out_path: Path, num_keep: int,
                          action_repr: str = "relative") -> dict:
    """Filter RAW dataset → only terminated episodes, drop last NaN frame, keep first num_keep.

    action_repr: 'relative' (default; ∈ [-1, 1], swm-env native)
                 'absolute' (target pixel = agent_pos + rel * 100, ∈ [0, 512], DP-paper convention)
    """
    assert action_repr in ("relative", "absolute"), f"unknown action_repr={action_repr}"
    print(f"[filter] reading {raw_path}")
    with h5py.File(raw_path, "r") as f:
        ep_len = f["ep_len"][:]
        ep_off = f["ep_offset"][:]
        terminated = f["terminated"][:]  # per-frame
        truncated  = f["truncated"][:]

        # An episode is a TIGHT success iff its LAST frame has terminated=True.
        success_idx = []
        for ep in range(len(ep_len)):
            o, L = int(ep_off[ep]), int(ep_len[ep])
            if L > 0 and bool(terminated[o + L - 1]):
                success_idx.append(ep)
            if len(success_idx) >= num_keep:
                break
        n_success_total = sum(
            1 for ep in range(len(ep_len))
            if int(ep_len[ep]) > 0 and bool(terminated[int(ep_off[ep]) + int(ep_len[ep]) - 1])
        )
        print(f"[filter] found {n_success_total} successes / {len(ep_len)} attempts"
              f"  (asked for {num_keep})")
        if len(success_idx) < num_keep:
            print(f"[filter] WARNING: only {len(success_idx)} successes — using all of them")

        # Build new ep_len/ep_offset (drop LAST FRAME of each kept ep to remove NaN)
        new_ep_len = np.array(
            [int(ep_len[i]) - 1 for i in success_idx], dtype=np.int32  # drop last
        )
        new_ep_off = np.zeros(len(new_ep_len), dtype=np.int64)
        new_ep_off[1:] = np.cumsum(new_ep_len[:-1])
        total_frames = int(new_ep_len.sum())
        print(f"[filter] keeping {len(success_idx)} eps, {total_frames} frames "
              f"(after dropping last NaN frame per ep)")

        # Per-frame datasets to copy (every dataset whose first dim equals total source frames)
        per_frame_keys = [
            k for k in f.keys()
            if k not in ("ep_len", "ep_offset")
            and f[k].ndim >= 1
            and f[k].shape[0] == f["action"].shape[0]
        ]
        print(f"[filter] copying keys: {per_frame_keys}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as g:
            g.create_dataset("ep_len", data=new_ep_len)
            g.create_dataset("ep_offset", data=new_ep_off)

            for key in per_frame_keys:
                src_d = f[key]
                shape = (total_frames,) + tuple(src_d.shape[1:])
                dst_d = g.create_dataset(
                    key, shape=shape, dtype=src_d.dtype,
                    chunks=(1,) + tuple(src_d.shape[1:]) if src_d.ndim > 1
                    else (min(1024, total_frames),),
                    compression="gzip" if key == "pixels" else None,
                )
                for new_ep, old_ep in enumerate(success_idx):
                    o_src = int(ep_off[old_ep])
                    L_old = int(ep_len[old_ep])
                    L_new = L_old - 1  # drop last
                    o_dst = int(new_ep_off[new_ep])
                    if L_new > 0:
                        dst_d[o_dst:o_dst+L_new] = src_d[o_src:o_src+L_new]
            # ep_idx is already in per_frame_keys (copied above with re-indexing intent below)
            # Re-index ep_idx so values map 0..len(success_idx)-1 in the new file.
            if "ep_idx" in g:
                new_ep_idx = np.concatenate(
                    [np.full(int(L), i, dtype=np.int32) for i, L in enumerate(new_ep_len)]
                )
                del g["ep_idx"]
                g.create_dataset("ep_idx", data=new_ep_idx)

    # action_repr='absolute': overwrite action column with abs pixel targets.
    # action_abs[t] = proprio[t, :2] + action_rel[t] * 100   (float64 to preserve precision)
    if action_repr == "absolute":
        with h5py.File(out_path, "r+") as g:
            rel = g["action"][:].astype(np.float64)
            agent_pos = g["proprio"][:, :2].astype(np.float64)
            abs_act = agent_pos + rel * 100.0
            del g["action"]
            g.create_dataset("action", data=abs_act, dtype="float64")
            g.attrs["action_repr"] = "absolute"
            g.attrs["action_scale"] = 100.0
        print(f"[finalize] rewrote action column as ABSOLUTE pixel target (float64). "
              f"Range: [{abs_act.min():.2f}, {abs_act.max():.2f}]")
    else:
        with h5py.File(out_path, "r+") as g:
            g.attrs["action_repr"] = "relative"
            g.attrs["action_scale"] = 100.0

    # Audit final dataset
    with h5py.File(out_path, "r") as g:
        a = g["action"][:]
        p = g["proprio"][:]
        bp = g["block_pose"][:]
        gp = g["goal_pose"][:]
        nan_a = int(np.isnan(a).sum())
        pos_err = np.linalg.norm(bp[:, :2] - gp[:, :2], axis=1)
        ang_err_raw = np.abs(bp[:, 2] - gp[:, 2])
        ang_err = np.minimum(ang_err_raw, 2*np.pi - ang_err_raw)
        # per-episode terminal
        ep_len_g = g["ep_len"][:]
        ep_off_g = g["ep_offset"][:]
        term_pos = []
        term_ang = []
        for i in range(len(ep_len_g)):
            L = int(ep_len_g[i])
            o = int(ep_off_g[i])
            term_pos.append(float(pos_err[o + L - 1]))
            term_ang.append(float(ang_err[o + L - 1]))
        dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
        # per-ep velocity (skip cross-episode boundaries)
        vel_per_step = []
        for i in range(len(ep_len_g)):
            L = int(ep_len_g[i])
            o = int(ep_off_g[i])
            if L >= 2:
                vel_per_step.extend(np.linalg.norm(np.diff(p[o:o+L], axis=0), axis=1).tolist())
        return {
            "n_eps": len(ep_len_g),
            "total_frames": int(np.sum(ep_len_g)),
            "ep_len_mean": float(np.mean(ep_len_g)),
            "ep_len_med":  float(np.median(ep_len_g)),
            "nan_actions": nan_a,
            "term_pos_med": float(np.median(term_pos)),
            "term_pos_p90": float(np.percentile(term_pos, 90)),
            "term_pos_max": float(np.max(term_pos)),
            "term_ang_med": float(np.median(term_ang)),
            "term_ang_p90": float(np.percentile(term_ang, 90)),
            "term_ang_max": float(np.max(term_ang)),
            "avg_velocity_per_step": float(np.mean(vel_per_step)),
            "med_velocity_per_step": float(np.median(vel_per_step)),
        }


@hydra.main(version_base=None, config_path="config", config_name="expert_pusht")
def run(cfg: DictConfig):
    cache_dir = os.path.join(os.environ.get("STABLEWM_HOME"), "pusht")
    dataset_name = cfg.dataset_name
    num_traj = cfg.num_traj
    max_steps = cfg.world.max_episode_steps
    slowdown = cfg.slowdown_factor
    seed = cfg.seed
    n_request = int(cfg.get("n_request", num_traj * 5))  # over-request to account for fail rate
    action_repr = str(cfg.get("action_repr", "relative"))  # 'relative' (default) or 'absolute'
    assert action_repr in ("relative", "absolute"), \
        f"action_repr must be 'relative' or 'absolute', got {action_repr}"

    raw_name = f"{dataset_name}_RAW"
    raw_path = Path(cache_dir) / "datasets" / f"{raw_name}.h5"
    if raw_path.exists():
        print(f"[collect] removing existing RAW file: {raw_path}")
        raw_path.unlink()

    # Optional damping for SlipperyPushT collection. cfg has no `damping` key by
    # default → None → plain (frictionless) PushT, IDENTICAL to the original
    # call. Pass `+damping=0.35` to collect slippery demos with the lerobot
    # expert; the plain pusht pipeline is unaffected.
    damping = cfg.get("damping", None)
    world = swm.World(
        "swm/PushT-v1",
        num_envs=cfg.world.num_envs,
        max_episode_steps=max_steps,
        image_shape=cfg.world.image_shape,
        history_size=1,
        frame_skip=1,
        damping=(None if damping is None else float(damping)),
    )
    print(f"[collect] env damping = {damping}")
    patch_pusht_tight_success(world)  # pos<5, ang<π/36

    # Optional narrow-init patch (Dataset B): narrows variation_space['block'] bounds
    narrow_init_cfg = cfg.get("narrow_init", None)
    if narrow_init_cfg is not None:
        from _pusht_narrow_init_patch import patch_pusht_narrow_init
        from omegaconf import OmegaConf as _OC
        ni_kwargs = _OC.to_container(narrow_init_cfg, resolve=True)
        patch_pusht_narrow_init(world, **ni_kwargs)
        print(f"[collect] narrow_init patched: {ni_kwargs}")

    policy = LeRobotDPPolicy(lr_cfg=cfg.lerobot, slowdown_factor=slowdown, device=cfg.device)
    world.set_policy(policy)

    print(f"[collect] Recording {n_request} attempts (slow={slowdown}, max_steps={max_steps}, TIGHT eval_state, "
          f"action_repr={action_repr}) -> {raw_name}")
    world.record_dataset(raw_name, episodes=n_request, seed=seed, cache_dir=cache_dir)

    # Post-process: filter, drop NaN, rename to final. final_out_dir defaults to
    # the original hardcoded path (plain pipeline unchanged); override with
    # `+final_out_dir=<dir>` for PACE / slippery collection.
    final_out_dir = cfg.get("final_out_dir", os.path.join(os.environ["STABLEWM_HOME"], "pusht/data"))
    final_path = Path(final_out_dir) / f"{dataset_name}.h5"
    if final_path.exists():
        print(f"[collect] removing existing FINAL file: {final_path}")
        final_path.unlink()
    stats = _filter_and_finalize(raw_path, final_path, num_keep=num_traj, action_repr=action_repr)

    print(f"\n=== FINAL DATASET STATS ({final_path}) ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<28} = {v:.4f}")
        else:
            print(f"  {k:<28} = {v}")
    print(f"\n[collect] final dataset: {final_path}  ({final_path.stat().st_size / 1e6:.1f} MB)")

    # Optionally clean up RAW file
    print(f"[collect] RAW file kept at {raw_path} (for re-filtering / debug). "
          f"Delete with: rm {raw_path}")


if __name__ == "__main__":
    run()
