"""Collect 1xDP_slippery_pusht_WM data: sticky DP on slippery env (damping=0.35).

Mirrors stable-worldmodel-srl/scripts/data/collect_weak_pusht.py but swaps the
WeakPolicy for a DP policy wrapper that batches inference across all envs.

Must run in an environment with the robomimic version the DP checkpoint needs
(the original collection used a separate robodiff conda environment).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import collections

import dill
import h5py
import hdf5plugin  # noqa: F401 — must precede h5py for blosc-lz4 compression
import hydra
import numpy as np
import torch
from loguru import logger as logging
from omegaconf import DictConfig

import stable_worldmodel as swm
from stable_worldmodel.policy import BasePolicy

# Ensure the DP repo is importable before Hydra's get_class resolves the
# workspace class stored in the checkpoint cfg._target_.
_DP_REPO = _REPO_ROOT / "diffusion_policy_repro" / "diffusion_policy"
if str(_DP_REPO) not in sys.path:
    sys.path.insert(0, str(_DP_REPO))


class DPCollectPolicy(BasePolicy):
    """Batched DP policy wrapper for swm.World.record_dataset.

    Loads the DP checkpoint, manages per-env n_obs_steps=2 history of
    (image, agent_pos), and a shared action queue.  On each get_action(infos)
    call:

      - extract current (image, agent_pos) per env from infos
      - detect resets via step_idx==0 or empty history
      - if queue empty: build (num_envs, n_obs_steps, ...) obs_dict, run
        policy.predict_action() once, store the resulting
        (num_envs, n_action_steps, 2) ABSOLUTE actions in the queue
      - pop one action timestep, convert to relative using live agent_pos:
            rel = (abs - live_agent_pos) / 100
      - return shape (num_envs, 2) float32
    """

    def __init__(
        self,
        ckpt_path: str,
        num_envs: int,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        device: str = "cuda:0",
        slowdown_factor: float = 1.0,
        pace_mode: str = "displacement",
    ):
        super().__init__()
        self.num_envs = int(num_envs)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.device = device
        # Pace knob. Two mechanisms (both bit-exact v1 at c=1.0):
        #   displacement: clip(rel,±1) * c — command inflation; measured to
        #     SATURATE for c>1 (queue still advances 1 target/step).
        #   stride: advance the abs-target queue at rate c — true time-rescale
        #     of the DP plan; speed ∝ c, labels stay within ±1.
        self.slowdown_factor = float(slowdown_factor)
        self.pace_mode = str(pace_mode)
        assert self.pace_mode in ("displacement", "stride"), self.pace_mode
        self._last_target = None
        self._stride_acc = 0.0

        self.policy = self._load_dp_policy(ckpt_path)

        # Per-env obs history
        self.img_history = [
            collections.deque(maxlen=self.n_obs_steps)
            for _ in range(self.num_envs)
        ]
        self.ap_history = [
            collections.deque(maxlen=self.n_obs_steps)
            for _ in range(self.num_envs)
        ]

        # Action queue: list of (num_envs, 2) ABSOLUTE pixel targets
        self.action_queue: list[np.ndarray] = []

    @staticmethod
    def _patch_robomimic_base_nets() -> None:
        """Inject CropRandomizer into robomimic.models.base_nets if missing.

        The robomimic-lewm fork moved CropRandomizer from base_nets to obs_core.
        The DP policy does `import robomimic.models.base_nets as rmbn` and uses
        rmbn.CropRandomizer, so we re-export it here before the workspace is
        instantiated.  This is a no-op if the attribute already exists.
        """
        import robomimic.models.base_nets as _rmbn
        if not hasattr(_rmbn, "CropRandomizer"):
            try:
                from robomimic.models.obs_core import CropRandomizer
                _rmbn.CropRandomizer = CropRandomizer
                logging.debug("Patched robomimic.models.base_nets.CropRandomizer from obs_core")
            except ImportError:
                pass  # nothing we can do; will fail at instantiation with a clear error

    def _load_dp_policy(self, ckpt_path: str):
        """Load DP checkpoint using the same pattern as eval.py."""
        self._patch_robomimic_base_nets()
        payload = torch.load(
            open(ckpt_path, "rb"), pickle_module=dill, map_location=self.device
        )
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.model
        if cfg.training.get("use_ema", False):
            policy = workspace.ema_model
        policy.to(self.device)
        policy.eval()
        logging.info(f"Loaded DP policy from {ckpt_path}")
        return policy

    def _extract_obs_per_env(
        self, infos: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read (num_envs, 3, H, W) float32 image and (num_envs, 2) agent_pos.

        infos["pixels"] shape: (num_envs, history_size, H, W, 3) uint8
        infos["state"]  shape: (num_envs, history_size, state_dim)
          where state[:, :, :2] is agent_pos in pixel coords [0, 512].
        """
        pixels = infos["pixels"]  # (N, history, H, W, 3) uint8
        if pixels.ndim == 5:
            pixels = pixels[:, -1]  # (N, H, W, 3)
        # The H5 stores these at world.image_shape (224, 224, matches pusht_expert_train).
        # DP checkpoint requires 96×96 input; downsample here for inference only.
        if pixels.shape[1] != 96 or pixels.shape[2] != 96:
            import cv2
            resized = np.stack(
                [cv2.resize(pixels[i], (96, 96), interpolation=cv2.INTER_AREA)
                 for i in range(pixels.shape[0])],
                axis=0,
            )
            pixels = resized
        # NHWC -> NCHW, float32 [0,1]
        imgs = np.transpose(pixels.astype(np.float32) / 255.0, (0, 3, 1, 2))

        state = infos["state"]  # (N, history, state_dim) or (N, state_dim)
        if state.ndim == 3:
            state = state[:, -1]  # (N, state_dim)
        pos = state[:, :2].astype(np.float32)  # agent xy in pixel coords
        return imgs, pos

    def _maybe_reset_history(
        self, infos: dict, imgs: np.ndarray, pos: np.ndarray
    ) -> None:
        """Re-init env history on episode reset (step_idx==0 or empty)."""
        step_idx = infos.get("step_idx", None)
        if step_idx is not None:
            # step_idx shape: (N,) or (N, history)
            si = np.asarray(step_idx)
            if si.ndim > 1:
                si = si[:, -1]
        else:
            si = None

        for i in range(self.num_envs):
            needs_reset = len(self.img_history[i]) == 0
            if si is not None and int(si[i]) == 0:
                needs_reset = True
            if needs_reset:
                self.img_history[i].clear()
                self.ap_history[i].clear()
                for _ in range(self.n_obs_steps):
                    self.img_history[i].append(imgs[i].copy())
                    self.ap_history[i].append(pos[i].copy())
                # Clear action queue on reset so we re-plan immediately
                self.action_queue = []
                self._last_target = None
                self._stride_acc = 0.0

    def _refill_queue(self) -> None:
        """Batched DP inference -> refill the shared ABS-target queue."""
        # (N, To, 3, H, W)
        img_stack = np.stack(
            [np.stack(list(h), axis=0) for h in self.img_history], axis=0
        )
        # (N, To, 2)
        ap_stack = np.stack(
            [np.stack(list(h), axis=0) for h in self.ap_history], axis=0
        )
        obs_dict = {
            "image": torch.from_numpy(img_stack.astype(np.float32)).to(
                self.device
            ),
            "agent_pos": torch.from_numpy(ap_stack.astype(np.float32)).to(
                self.device
            ),
        }
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)
        abs_actions = (
            action_dict["action"].detach().cpu().numpy()
        )  # (N, Ta, 2)
        self.action_queue = [
            abs_actions[:, t, :].copy()
            for t in range(abs_actions.shape[1])
        ]

    def get_action(self, infos: dict, **kwargs) -> np.ndarray:
        imgs, pos = self._extract_obs_per_env(infos)
        self._maybe_reset_history(infos, imgs, pos)

        # Append current frame to each history
        for i in range(self.num_envs):
            self.img_history[i].append(imgs[i])
            self.ap_history[i].append(pos[i])

        if self.pace_mode == "stride":
            # TRUE time-rescale: advance the ABS-target queue at rate c
            # (fractional accumulator), track the current target with plain
            # clipped rel. c>1 skips intermediate targets (SuP-decimation-like),
            # c<1 holds each target for ~1/c steps. c=1.0 == displacement mode
            # == v1 pipeline bit-exactly. No displacement multiplier: speed
            # comes from the plan clock, not from command inflation.
            self._stride_acc += self.slowdown_factor
            n_pop = int(self._stride_acc)
            self._stride_acc -= n_pop
            if self._last_target is None and n_pop == 0:
                n_pop = 1  # first call after reset at c<1: must acquire a target
            for _ in range(n_pop):
                if len(self.action_queue) == 0:
                    self._refill_queue()
                self._last_target = self.action_queue.pop(0)
            abs_step = self._last_target  # (N, 2)
            rel = ((abs_step - pos) / 100.0).astype(np.float32)
            rel = np.clip(rel, -1.0, 1.0)
            return rel

        # displacement mode (v1-compatible): pop one target per step, scale the
        # clipped rel by c (afterclip). NOTE measured caveat: c>1 saturates
        # (~1.25x at c=2.5) because the queue still advances 1 target/step.
        if len(self.action_queue) == 0:
            self._refill_queue()
        abs_step = self.action_queue.pop(0)  # (N, 2)
        rel = ((abs_step - pos) / 100.0).astype(np.float32)
        rel = np.clip(rel, -1.0, 1.0)
        if self.slowdown_factor != 1.0:
            rel = (rel * self.slowdown_factor).astype(np.float32)
        return rel


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="dp_slippery_pusht_play",
)
def run(cfg: DictConfig):
    # damping=null -> plain PushT physics (env default); a float -> slippery.
    damping = cfg.get("damping", None)
    world = swm.World(
        "swm/PushT-v1",
        damping=(float(damping) if damping is not None else None),
        **cfg.world,
    )

    policy = DPCollectPolicy(
        ckpt_path=cfg.dp_ckpt_path,
        num_envs=int(cfg.world.num_envs),
        n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=int(cfg.n_action_steps),
        device=cfg.device,
        slowdown_factor=float(cfg.get("slowdown_factor", 1.0)),
        pace_mode=str(cfg.get("pace_mode", "displacement")),
    )
    world.set_policy(policy)

    traj_per_shard = cfg.num_traj // cfg.num_shards
    rng = np.random.default_rng(cfg.seed)
    shard_offset = int(cfg.get("shard_offset", 0))

    from pathlib import Path
    out_dir = Path(cfg.cache_dir) / "datasets" / cfg.dataset_name
    for i in range(cfg.num_shards):
        shard_idx = shard_offset + i
        shard_path = out_dir / f"shard_{shard_idx}.h5"
        if shard_path.exists():
            logging.warning(
                f"shard_{shard_idx}.h5 already exists at {shard_path} — record_dataset "
                f"will APPEND. To force a fresh write, delete it first."
            )
        world.record_dataset(
            f"{cfg.dataset_name}/shard_{shard_idx}",
            episodes=traj_per_shard,
            seed=rng.integers(0, 1_000_000).item(),
            cache_dir=cfg.cache_dir,
            options=cfg.get("options"),
        )
        # Prune shard to match pusht_expert_train.h5 schema exactly:
        # 8 keys (action, ep_len, ep_offset, episode_idx, pixels, proprio, state, step_idx),
        # proprio/state as float32, ep_idx renamed to episode_idx, pixels gzip-compressed.
        _prune_shard_to_expert_schema(
            shard_path,
            damping=(float(damping) if damping is not None else float("nan")),
            policy_source=str(cfg.dp_ckpt_path),
            slowdown_factor=float(cfg.get("slowdown_factor", 1.0)),
        )

    logging.success(
        f"Collected {cfg.num_traj} eps to {cfg.cache_dir}/datasets/{cfg.dataset_name}/"
    )


def _prune_shard_to_expert_schema(
    shard_path: "Path", damping: float, policy_source: str,
    slowdown_factor: float = 1.0,
) -> None:
    """Rewrite the shard h5 to the expert 8-key schema + 4 audit keys.

    Kept (matching pusht_expert_train.h5 dtypes):
      action       (N, 2)        float32
      ep_len       (E,)          int32
      ep_offset    (E,)          int64
      episode_idx  (N,)          int64    (renamed from swm's `ep_idx`)
      pixels       (N, H, W, 3)  uint8    (gzip)
      proprio      (N, 4)        float32
      state        (N, 7)        float32
      step_idx     (N,)          int64

    Kept IF PRESENT (audit keys — v1's prune dropped these, which cost us:
    no success flag forced state-rescoring audits, and concat's goal_pose
    0.0-fill produced the origin-distance landmine):
      terminated   (N,)  bool     truncated  (N,)  bool
      goal_pose    (N,*) float32  coverage   (N,)  float32

    Dropped: block_pose, env_name, goal, goal_proprio, goal_state, id,
      n_contacts, pos_agent, render_time, reward, score, vel_agent.

    Adds trace attrs (damping [NaN = plain physics], policy_source,
    dataset_kind, slowdown_factor).
    """
    import shutil
    from pathlib import Path
    shard_path = Path(shard_path)
    tmp_path = shard_path.with_suffix(".pruned.h5")

    kept = {
        "action":     np.float32,
        "ep_len":     np.int32,
        "ep_offset":  np.int64,
        "episode_idx": np.int64,   # renamed from `ep_idx`
        "pixels":     np.uint8,
        "proprio":    np.float32,
        "state":      np.float32,
        "step_idx":   np.int64,
    }
    kept_optional = {
        "terminated": np.bool_,
        "truncated":  np.bool_,
        "goal_pose":  np.float32,
        "coverage":   np.float32,
    }
    with h5py.File(shard_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        for out_key, out_dtype in kept.items():
            src_key = "ep_idx" if out_key == "episode_idx" else out_key
            if src_key not in src:
                raise KeyError(f"required key '{src_key}' missing in {shard_path}")
            data = src[src_key][:].astype(out_dtype)
            if out_key == "pixels":
                dst.create_dataset(
                    out_key, data=data, dtype=out_dtype,
                    chunks=(1, *data.shape[1:]), compression="gzip",
                )
            else:
                dst.create_dataset(out_key, data=data, dtype=out_dtype)
        for out_key, out_dtype in kept_optional.items():
            if out_key in src:
                dst.create_dataset(
                    out_key, data=src[out_key][:].astype(out_dtype), dtype=out_dtype
                )
            else:
                logging.warning(
                    f"optional audit key '{out_key}' absent in {shard_path} — skipped"
                )
        dst.attrs["damping"] = float(damping)
        dst.attrs["policy_source"] = str(policy_source)
        dst.attrs["dataset_kind"] = "play"
        dst.attrs["slowdown_factor"] = float(slowdown_factor)
    shutil.move(str(tmp_path), str(shard_path))


if __name__ == "__main__":
    run()
