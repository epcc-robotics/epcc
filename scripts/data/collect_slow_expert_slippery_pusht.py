"""Collect 0.5x slow slippery PushT expert demos using our locally-trained slippery DP.

Composes pieces already in this directory:
  - DPCollectPolicy (slippery DP loader + batched inference) from
    collect_dp_slippery_pusht_play.py — subclassed here to add slowdown_factor.
  - patch_pusht_tight_success + _filter_and_finalize (tight eval_state + post-process)
    from collect_slow_expert_pusht_tight.py.

Output schema = 24-key raw swm (matches $STABLEWM_HOME/pusht/data/
expert_slow_050_n200.h5 style), with attrs action_repr='relative', action_scale=100.0.

Usage:
    MUJOCO_GL=egl python \\
        scripts/data/collect_slow_expert_slippery_pusht.py \\
        world.num_envs=25 \\
        world.max_episode_steps=500 \\
        slowdown_factor=0.5 \\
        num_traj=200 \\
        n_request=800 \\
        damping=0.35 \\
        dataset_name=expert_slow_050_n200_d035 \\
        seed=3072
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

# Make sibling collect_*.py files importable as bare modules
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
# Repo root (for any cross-package imports inside the dependencies)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hdf5plugin  # noqa: F401 — must precede h5py
import h5py  # noqa: F401
import hydra
import numpy as np
import torch
import stable_worldmodel as swm
from omegaconf import DictConfig

from collect_dp_slippery_pusht_play import DPCollectPolicy
from collect_slow_expert_pusht_tight import (
    patch_pusht_tight_success,
    _filter_and_finalize,
)


class SlowSlipperyDPPolicy(DPCollectPolicy):
    """Slippery DP + slowdown_factor applied to the abs->rel conversion.

    Inherits all DP loading / batched inference / per-env history logic from
    DPCollectPolicy and overrides get_action to multiply the relative action
    by `slowdown_factor` before clipping to [-1, 1].
    """

    def __init__(self, *args, slowdown_factor: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.slowdown_factor = float(slowdown_factor)

    def get_action(self, infos: dict, **kwargs) -> np.ndarray:
        imgs, pos = self._extract_obs_per_env(infos)
        self._maybe_reset_history(infos, imgs, pos)

        for i in range(self.num_envs):
            self.img_history[i].append(imgs[i])
            self.ap_history[i].append(pos[i])

        if len(self.action_queue) == 0:
            img_stack = np.stack(
                [np.stack(list(h), axis=0) for h in self.img_history], axis=0
            )
            ap_stack = np.stack(
                [np.stack(list(h), axis=0) for h in self.ap_history], axis=0
            )
            obs_dict = {
                "image": torch.from_numpy(img_stack.astype(np.float32)).to(self.device),
                "agent_pos": torch.from_numpy(ap_stack.astype(np.float32)).to(self.device),
            }
            with torch.no_grad():
                action_dict = self.policy.predict_action(obs_dict)
            abs_actions = action_dict["action"].detach().cpu().numpy()  # (N, Ta, 2)
            self.action_queue = [
                abs_actions[:, t, :].copy() for t in range(abs_actions.shape[1])
            ]

        abs_step = self.action_queue.pop(0)  # (N, 2)
        rel = ((abs_step - pos) / 100.0 * self.slowdown_factor).astype(np.float32)
        rel = np.clip(rel, -1.0, 1.0)
        return rel


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="dp_slippery_pusht_slow_expert",
)
def run(cfg: DictConfig):
    cache_dir = str(cfg.cache_dir)
    dataset_name = str(cfg.dataset_name)
    num_traj = int(cfg.num_traj)
    max_steps = int(cfg.world.max_episode_steps)
    slowdown = float(cfg.slowdown_factor)
    seed = int(cfg.seed)
    n_request = int(cfg.get("n_request", num_traj * 4))

    raw_name = f"{dataset_name}_RAW"
    raw_path = Path(cache_dir) / "datasets" / f"{raw_name}.h5"
    if raw_path.exists():
        print(f"[collect] removing existing RAW file: {raw_path}")
        raw_path.unlink()

    world = swm.World(
        "swm/PushT-v1",
        damping=float(cfg.damping),
        num_envs=int(cfg.world.num_envs),
        image_shape=list(cfg.world.image_shape),
        history_size=int(cfg.world.history_size),
        frame_skip=int(cfg.world.frame_skip),
        max_episode_steps=max_steps,
        verbose=int(cfg.world.get("verbose", 0)),
        render_mode=str(cfg.world.get("render_mode", "rgb_array")),
    )
    patch_pusht_tight_success(world)  # pos<5, ang<π/36

    # Canonical-init injection (Ablation A 1+2): serve fixed init states from an
    # npy list, each repeated init_repeats times (retry budget); every env reset
    # pops the next queue entry via options['state'] (7-dim; velocity zeroed by
    # _set_state). n_request MUST equal len(list)*init_repeats.
    init_npy = cfg.get("init_npy", None)
    if init_npy:
        arr = np.load(str(init_npy)); R = int(cfg.get("init_repeats", 5))
        _queue = iter([arr[i] for i in range(len(arr)) for _ in range(R)])
        n_patched = 0
        for _env in world.envs.unwrapped.envs:
            _base = _env
            while hasattr(_base, "env"):
                _base = _base.env
            _orig = _base.reset
            def _inj_reset(*a, _o=_orig, seed=None, options=None, **kw):
                try:
                    _st = next(_queue)
                except StopIteration:
                    _st = arr[0]
                options = dict(options or {})
                options["state"] = np.asarray(_st, dtype=np.float64)
                return _o(seed=seed, options=options, **kw)
            _base.reset = _inj_reset
            n_patched += 1
        print(f"[collect] canonical-init injection: {len(arr)} inits x {R} tries ({n_patched} envs)")

    # Ablation A OOD-init: restrict BLOCK inits to the outer ring
    # L_inf(init, center 256) >= ring, i.e. [100,400]^2 minus the central box.
    # The WM play corpus concentrates in the central basin (outer-ring occupancy
    # <1%/cell), so ring inits start demos deep in thin corpus support.
    # Implemented via the Box's built-in constrain_fn rejection sampling; only
    # the init path resamples block.start_position (goal_state is fixed by the
    # World wrapper), so goals stay at the canonical (256, 256, pi/4).
    ring = float(cfg.get("init_block_ring", 0.0))
    if ring > 0:
        def _ring_ok(v, _r=ring):
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            return bool(max(abs(v[0] - 256.0), abs(v[1] - 256.0)) >= _r)
        n_patched = 0
        for env in world.envs.unwrapped.envs:
            base = env
            while hasattr(base, "env"):
                base = base.env
            base.variation_space["block"]["start_position"].constrain_fn = _ring_ok
            n_patched += 1
        print(f"[collect] OOD-init ring patch: block |init-256|_inf >= {ring} "
              f"({n_patched} envs)")

    policy = SlowSlipperyDPPolicy(
        ckpt_path=str(cfg.dp_ckpt_path),
        num_envs=int(cfg.world.num_envs),
        n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=int(cfg.n_action_steps),
        device=str(cfg.device),
        slowdown_factor=slowdown,
    )
    world.set_policy(policy)

    print(
        f"[collect] Recording {n_request} attempts "
        f"(slow={slowdown}, max_steps={max_steps}, damping={cfg.damping}, "
        f"TIGHT eval_state, num_envs={cfg.world.num_envs}) -> {raw_name}"
    )
    world.record_dataset(raw_name, episodes=n_request, seed=seed, cache_dir=cache_dir)

    final_path = Path(str(cfg.final_out_dir)) / f"{dataset_name}.h5"
    if final_path.exists():
        print(f"[collect] removing existing FINAL file: {final_path}")
        final_path.unlink()
    stats = _filter_and_finalize(
        raw_path, final_path, num_keep=num_traj, action_repr="relative"
    )

    print(f"\n=== FINAL DATASET STATS ({final_path}) ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<28} = {v:.4f}")
        else:
            print(f"  {k:<28} = {v}")
    print(
        f"\n[collect] final dataset: {final_path}  "
        f"({final_path.stat().st_size / 1e6:.1f} MB)"
    )
    print(
        f"[collect] RAW file kept at {raw_path} (for re-filtering / debug). "
        f"Delete with: rm {raw_path}"
    )


if __name__ == "__main__":
    run()
