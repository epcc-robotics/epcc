"""Eval entry for CleanDiffuser DP ckpts on our OGBench cube AW env.

Loads a CleanDiffuser-trained DP agent (saved by `Logger.save_agent` as
``{model, model_ema}`` state dict) and runs rollouts via the same env loop
/ ring-buffer / RolloutLogger plumbing as eval_dp.py.

Submit example:
    DIFF_CKPT=$STABLEWM_HOME/checkpoints/dp/cube_h128/model_90000.pt \
    TRAIN_H5=$STABLEWM_HOME/datasets/cube_multiview_diffusion_c0.3_postpad_fullscale.h5 \
    POLICY=pv_aw_cube_scratch_wm_training/lewm_pv_aw_cube_scratch_epoch_11 \
    DATASET="Multiview WM/cube_multiview_diffusion_c0.3_postpad_fullscale" \
    N_EVAL=10 RUN_NAME=eval_cd_dp_n10 \
        sbatch submit_eval_cleandiffuser_dp_cube_aw.sh
"""
from __future__ import annotations

import os
import sys
import time
import collections
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

# CleanDiffuser checkout — on PYTHONPATH via sbatch.
from cleandiffuser.diffusion.ddpm import DDPM
from cleandiffuser.nn_condition import MultiImageObsCondition
from cleandiffuser.nn_diffusion import ChiUNet1d, DiT1d, ChiTransformer
from cleandiffuser.dataset.dataset_utils import MinMaxNormalizer

# Reuse env / world / policy infrastructure from eval_dp.py / mpc_multiview.
sys.path.insert(0, str(Path(__file__).parent.resolve()))
import stable_worldmodel as swm  # noqa: E402
from eval_mpc_multiview import (  # noqa: E402
    MultiviewWorldModelPolicy,
    _setup_eval_world,
    evaluate_tracking_from_dataset,
    _attach_wrist_renderers,
    _attach_proprio_alias,
)
from eval_mpcc_multiview import _setup_eval_solver_mpcc  # noqa: E402
from latent_modules.mpcc.reference import ReferenceTrajectory  # noqa: E402

try:
    import hdf5plugin  # noqa: F401 — register BLOSC/LZF for compressed H5
except ImportError:
    pass
import h5py


# =============================================================================
# Action MinMaxNormalizer — re-fit from training H5 (lightweight, ~1s)
# =============================================================================

def _load_action_minmax_from_h5(h5_path: str) -> MinMaxNormalizer:
    print(f"[cd-eval] computing action MinMax from {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        action = f["action"][:].astype(np.float32)
    action = np.nan_to_num(action, nan=0.0)
    norm = MinMaxNormalizer(action)
    print(f"[cd-eval] action min={norm.min.tolist()} max={norm.max.tolist()}")
    return norm


def _load_action_minmax(ckpt_path: str, train_h5_fallback: str = "") -> MinMaxNormalizer:
    """Load action MinMax — sibling normalizer_stats.npz first, then train H5.

    Layout: ckpt = .../<exp>/42/models/model_<step>.pt
                            └ work_dir = .../<exp>/42
                              └ normalizer_stats.npz (dumped by dp_image.py)
    """
    from pathlib import Path as _P
    work_dir = _P(ckpt_path).parent.parent
    stats_path = work_dir / "normalizer_stats.npz"
    if stats_path.is_file():
        print(f"[cd-eval] loading action MinMax from {stats_path}")
        s = np.load(stats_path)
        norm = MinMaxNormalizer.__new__(MinMaxNormalizer)
        norm.min = np.asarray(s["action_min"], dtype=np.float32)
        norm.max = np.asarray(s["action_max"], dtype=np.float32)
        norm.range = norm.max - norm.min
        if np.any(norm.range == 0):
            norm.range[norm.range == 0] = 1.0
        print(f"[cd-eval] action min={norm.min.tolist()} max={norm.max.tolist()}")
        return norm
    if not train_h5_fallback:
        raise FileNotFoundError(
            f"No {stats_path} and TRAIN_H5 fallback not provided. "
            f"Either dump normalizer_stats.npz alongside the ckpt or set "
            f"+diffusion.train_h5_path=/path/to/train.h5"
        )
    print(f"[cd-eval] {stats_path} not found; falling back to train H5")
    return _load_action_minmax_from_h5(train_h5_fallback)


# =============================================================================
# Pixel preprocessing — env render -> [-1, +1] tensor for MultiImageObsCondition
# =============================================================================

def _world_pixels_to_cd_tensor(
    info_dict: dict,
    *,
    use_wrist: bool,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Pull current pixel(s) from world.infos and convert to CleanDiffuser
    cond-ready tensor of shape (N_env, V, C, H, W) float32 in [-1, +1].

    Chain: uint8 (H, W, C) -> /255 -> ImageNormalizer (x*2-1) -> [-1, +1].
    """
    def _coerce(p):
        if hasattr(p, "cpu"):
            p = p.cpu().numpy()
        p = np.asarray(p)
        if p.ndim == 5:                       # (N_env, T, ...) -> last frame
            p = p[:, -1]
        if p.shape[-1] in (1, 3, 4) and p.shape[-3] not in (1, 3, 4):
            p = np.transpose(p, (0, 3, 1, 2))  # NHWC -> NCHW
        return torch.from_numpy(p).float().div(255.0)

    agent = _coerce(info_dict["pixels"]).to(device)
    views = [agent]
    if use_wrist and "wrist_pixels" in info_dict:
        views.append(_coerce(info_dict["wrist_pixels"]).to(device))
    x = torch.stack(views, dim=1)             # (N_env, V, C, H, W)

    if x.shape[-1] != image_size:
        N, V, C, H, W = x.shape
        x = x.reshape(N * V, C, H, W)
        x = F.interpolate(x, size=(image_size, image_size),
                          mode="bilinear", align_corners=False)
        x = x.reshape(N, V, C, image_size, image_size)

    x = x * 2.0 - 1.0                          # ImageNormalizer: [0,1] -> [-1, +1]
    return x.contiguous()


# =============================================================================
# CleanDiffuser DP planner — bundle (agent + action_norm + meta)
# =============================================================================

class CleanDiffuserDPPlanner:
    def __init__(self, *, agent, action_norm, horizon, action_dim,
                 obs_steps, action_steps, image_size, n_views,
                 sample_steps, device):
        self.agent = agent
        self.action_norm = action_norm
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.obs_steps = int(obs_steps)
        self.action_steps = int(action_steps)
        self.image_size = int(image_size)
        self.n_views = int(n_views)
        self.sample_steps = int(sample_steps)
        self.device = device
        # Naming aliases for parity with FromScratchDPPlanner consumers.
        self.Tp = self.horizon
        self.K = self.obs_steps
        self.stride = 1
        self.out_dim = self.action_dim


def build_cleandiffuser_dp_planner(
    ckpt_path: str,
    train_h5_path: str,
    cfg: DictConfig,
    device: str = "cuda",
) -> CleanDiffuserDPPlanner:
    image_size = int(cfg.diffusion.get("image_size", 96))
    n_views = int(cfg.diffusion.get("n_views", 2))
    action_dim = int(cfg.diffusion.get("action_dim", 5))
    horizon = int(cfg.diffusion.get("horizon", 16))
    obs_steps = int(cfg.diffusion.get("obs_steps", 2))
    action_steps = int(cfg.diffusion.get("action_steps", 8))
    crop_shape = list(cfg.diffusion.get("crop_shape", [84, 84]))
    use_group_norm = bool(cfg.diffusion.get("use_group_norm", True))
    use_seq = bool(cfg.diffusion.get("use_seq", True))
    sample_steps = int(cfg.diffusion.get("sample_steps", 50))
    nn_kind = str(cfg.diffusion.get("nn", "chi_unet"))

    # Single-view (pusht SV, key `image`) vs multi-view (cube AW, agentview+wrist).
    # Must match the training shape_meta so state_dict keys align. [ported from le-wm]
    if n_views == 1:
        obs_meta = {"image": {"shape": [3, image_size, image_size], "type": "rgb"}}
    else:
        obs_meta = {
            "agentview_image": {"shape": [3, image_size, image_size], "type": "rgb"},
            "wrist_image":     {"shape": [3, image_size, image_size], "type": "rgb"},
        }
    shape_meta = {"action": {"shape": [action_dim]}, "obs": obs_meta}

    nn_condition = MultiImageObsCondition(
        shape_meta=shape_meta, emb_dim=256,
        rgb_model_name="resnet18", resize_shape=None,
        crop_shape=crop_shape, random_crop=True,   # center-crops in eval mode
        use_group_norm=use_group_norm, use_seq=use_seq,
    ).to(device)

    if nn_kind == "chi_unet":
        nn_diffusion = ChiUNet1d(
            action_dim, 256, obs_steps,
            model_dim=256, emb_dim=256, dim_mult=[1, 2, 2],
            obs_as_global_cond=True, timestep_emb_type="positional",
        ).to(device)
    elif nn_kind == "dit":
        nn_diffusion = DiT1d(
            action_dim, emb_dim=256 * obs_steps,
            d_model=320, n_heads=10, depth=2, timestep_emb_type="fourier",
        ).to(device)
    elif nn_kind == "chi_transformer":
        nn_diffusion = ChiTransformer(
            action_dim, 256, horizon, obs_steps,
            d_model=256, nhead=4, num_layers=4, timestep_emb_type="positional",
        ).to(device)
    else:
        raise ValueError(f"unsupported nn kind: {nn_kind}")

    x_max = torch.ones((1, horizon, action_dim), device=device) * +1.0
    x_min = torch.ones((1, horizon, action_dim), device=device) * -1.0
    agent = DDPM(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition,
        device=device, diffusion_steps=sample_steps,
        x_max=x_max, x_min=x_min,
    )
    print(f"[cd-eval] loading agent from {ckpt_path}")
    agent.load(ckpt_path)
    agent.model.eval()
    agent.model_ema.eval()

    action_norm = _load_action_minmax(ckpt_path, train_h5_fallback=train_h5_path)
    return CleanDiffuserDPPlanner(
        agent=agent, action_norm=action_norm,
        horizon=horizon, action_dim=action_dim,
        obs_steps=obs_steps, action_steps=action_steps,
        image_size=image_size, n_views=n_views,
        sample_steps=sample_steps, device=device,
    )


# =============================================================================
# Batched sample fn
# =============================================================================

@torch.no_grad()
def dp_cleandiffuser_sample_actions_batched(
    planner: CleanDiffuserDPPlanner,
    history_pixels: torch.Tensor,    # (B, K, V, C, H, W), [-1, +1] float32
) -> torch.Tensor:
    B, K, V = history_pixels.shape[:3]
    assert K == planner.K and V == planner.n_views, \
        f"history mismatch: K={K}/{planner.K}, V={V}/{planner.n_views}"
    history_pixels = history_pixels.to(planner.device, non_blocking=True)

    if planner.n_views == 1:
        cond_dict = {"image": history_pixels[:, :, 0].contiguous()}  # (B, K, C, H, W)
    else:
        cond_dict = {
            "agentview_image": history_pixels[:, :, 0].contiguous(),  # (B, K, C, H, W)
            "wrist_image":     history_pixels[:, :, 1].contiguous(),
        }

    prior = torch.zeros(B, planner.horizon, planner.action_dim, device=planner.device)
    traj_norm, _ = planner.agent.sample(
        prior, n_samples=B,
        sample_steps=planner.sample_steps,
        use_ema=True, temperature=1.0,
        condition_cfg=cond_dict, w_cfg=1.0,
    )    # (B, horizon, action_dim) in [-1, +1]

    traj_np = traj_norm.detach().cpu().numpy().reshape(-1, planner.action_dim)
    actions = planner.action_norm.unnormalize(traj_np).reshape(
        B, planner.horizon, planner.action_dim
    )
    return torch.from_numpy(actions).float().to(planner.device).contiguous()


# =============================================================================
# CleanDiffuser DP Policy — ring buffer of K=2 pixel frames
# =============================================================================

class CleanDiffuserDPPolicy(MultiviewWorldModelPolicy):
    def __init__(self, *, dp: CleanDiffuserDPPlanner,
                 action_block: int, receding_horizon: int,
                 dump_dir: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._dp = dp
        self._action_block = int(action_block)
        self._buffer_max = int(action_block) * int(receding_horizon)
        if self._buffer_max > dp.horizon:
            raise ValueError(
                f"action_block*receding_horizon ({self._buffer_max}) > horizon "
                f"({dp.horizon})"
            )
        self._buffer: collections.deque = collections.deque()
        self._ring_len = dp.K               # K=2, stride=1
        self._ring: torch.Tensor | None = None  # (N_env, K, V, C, H, W)
        self._stride_indices = list(range(dp.K))
        self._n_replans = 0
        self._n_ring_updates = 0
        self._dump_dir = dump_dir
        if self._dump_dir is not None:
            os.makedirs(self._dump_dir, exist_ok=True)
        self._env_step = 0

    def _update_ring(self, cur_pixels: torch.Tensor) -> None:
        if self._ring is None:
            self._ring = (
                cur_pixels.unsqueeze(1)
                .repeat(1, self._ring_len, 1, 1, 1, 1)
                .contiguous()
            )
        else:
            self._ring = torch.roll(self._ring, shifts=-1, dims=1)
            self._ring[:, -1] = cur_pixels
        self._n_ring_updates += 1

    def _sample_actions(self) -> torch.Tensor:
        history_K = self._ring[:, self._stride_indices]
        actions = dp_cleandiffuser_sample_actions_batched(self._dp, history_K)
        if actions.shape[1] < self._buffer_max:
            raise RuntimeError(
                f"CD DP produces {actions.shape[1]} actions/replan but "
                f"buffer_max={self._buffer_max}"
            )
        return actions[:, :self._buffer_max, :].contiguous()

    def get_action(self, info_dict, **kwargs):
        cur_pixels = _world_pixels_to_cd_tensor(
            info_dict,
            use_wrist=self._dp.n_views > 1,
            image_size=self._dp.image_size,
            device=self._dp.device,
        )
        self._update_ring(cur_pixels)
        self._env_step += 1

        if len(self._buffer) == 0:
            actions = self._sample_actions()
            if self._dump_dir is not None:
                fp = os.path.join(self._dump_dir, f"replan_{self._n_replans:03d}.pt")
                torch.save({
                    "history_K": self._ring[:, self._stride_indices].cpu(),
                    "actions": actions.cpu(),
                    "env_step": self._env_step,
                }, fp)
            self._n_replans += 1
            for t in range(actions.shape[1]):
                self._buffer.append(actions[:, t])
        a = self._buffer.popleft()
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        return a


# =============================================================================
# Hydra main — mirrors eval_dp.py run() with CleanDiffuser planner/policy
# =============================================================================

@hydra.main(version_base=None, config_path="./config/eval", config_name="multiview_cube_mpcc")
def run(cfg: DictConfig):
    diff_cfg = cfg.get("diffusion", None)
    assert diff_cfg is not None and diff_cfg.get("ckpt_path"), (
        "cfg.diffusion.ckpt_path required "
        "(Hydra: +diffusion.ckpt_path=/path/to/model_*.pt)"
    )
    # train_h5_path is the FALLBACK for normalizer stats when the ckpt's
    # sibling normalizer_stats.npz is missing (legacy ckpts trained before
    # 2026-05-21). New ckpts dump stats alongside themselves — TRAIN_H5
    # may be omitted entirely. See _load_action_minmax in this file.
    train_h5_path = str(diff_cfg.get("train_h5_path", ""))
    if not train_h5_path:
        # Auto-guess from STABLEWM_HOME + dataset_name — best-effort fallback.
        candidate = os.path.join(
            os.environ.get("STABLEWM_HOME", ""),
            f"{cfg.eval.dataset_name}.h5",
        )
        if os.path.isfile(candidate):
            train_h5_path = candidate
    # Don't assert here — _load_action_minmax raises only if BOTH the
    # sibling .npz and train_h5_path are missing.

    # -- Phase 1: env + dataset --
    eval_world = _setup_eval_world(cfg)

    # -- Phase 2: build planner --
    dp = build_cleandiffuser_dp_planner(
        ckpt_path=str(diff_cfg.ckpt_path),
        train_h5_path=train_h5_path,
        cfg=cfg, device="cuda",
    )

    plan_action_block = int(cfg.plan_config.action_block)
    receding_horizon = int(cfg.plan_config.receding_horizon)
    n_actions_needed = plan_action_block * receding_horizon
    if n_actions_needed > dp.horizon:
        raise ValueError(
            f"action_block × receding_horizon ({n_actions_needed}) > horizon "
            f"({dp.horizon}); CleanDiffuser chunks have {dp.horizon} env-actions."
        )
    print(f"[cd-eval] horizon={dp.horizon}  K={dp.K}  action_dim={dp.action_dim}  "
          f"  actions/replan={n_actions_needed}")

    # -- Phase 3: solver + Policy --
    n_eval = eval_world.n_eval
    _policy_dir_pre = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
    _run_name_pre = cfg.output.get("run_name") or "eval_cleandiffuser_dp"
    _dump_dir_pre = str(_policy_dir_pre / _run_name_pre / "replan_dumps")

    model, solver, policy, plan_config = _setup_eval_solver_mpcc(
        cfg, eval_world,
        gt_refs=False,
        policy_cls=CleanDiffuserDPPolicy,
        policy_kwargs={
            "dp": dp,
            "action_block": plan_action_block,
            "receding_horizon": receding_horizon,
            "dump_dir": _dump_dir_pre,
        },
    )

    # Standard cube WM patch (HF ViT 4.46.x compat) — mirror eval_dp.py.
    import transformers as _hf_transformers
    for _m in model.modules():
        if hasattr(_m, "config") and isinstance(_m.config, _hf_transformers.PretrainedConfig):
            _fresh = type(_m.config)()
            for _k, _v in _fresh.__dict__.items():
                if _k not in _m.config.__dict__:
                    setattr(_m.config, _k, _v)
        if "ViTSelfAttention" in type(_m).__name__ and not hasattr(_m, "dropout"):
            _m.dropout = torch.nn.Dropout(getattr(_m, "dropout_prob", 0.0))

    # Dummy refs for cost-model init path (policy bypasses solver.solve).
    _D_for_dummy = int(getattr(model, "emb_dim", None)
                       or getattr(getattr(model, "config", None), "emb_dim", None)
                       or 384)
    _dummy_emb = torch.zeros(2, _D_for_dummy)
    solver._references = [
        ReferenceTrajectory.from_latents(_dummy_emb.clone()) for _ in range(n_eval)
    ]
    solver._env_ref_indices = np.arange(n_eval)

    # -- Phase 4: bind policy + renderer + proprio --
    eval_world.world.set_policy(policy)
    enabled_views = getattr(model, "enabled_views", ("agent", "wrist"))
    proprio_keys = tuple(getattr(model, "proprio_keys", ()) or ())
    if "wrist" in enabled_views:
        _attach_wrist_renderers(eval_world.world, img_size=cfg.world.get("width", 224))
        print(f"[renderer] wrist renderer attached")
    if "proprio" in enabled_views:
        _attach_proprio_alias(eval_world.world, proprio_keys)

    # -- Phase 5: output path --
    results_path = _policy_dir_pre / _run_name_pre
    results_path.mkdir(parents=True, exist_ok=True)
    print(
        f"[cd-eval] episodes={np.unique(eval_world.eval_episodes).tolist()}, "
        f"goal_offset={eval_world.goal_offset}, eval_budget={eval_world.eval_budget}, "
        f"n={n_eval}, dp_ckpt={Path(diff_cfg.ckpt_path).name}"
    )

    # -- Phase 6: RolloutLogger + evaluate loop --
    callables = (OmegaConf.to_container(cfg.eval.callables, resolve=True)
                 if cfg.eval.get("callables") else None)

    rollout_logger = None
    if cfg.eval.get("rollout_log", False):
        from latent_modules.eval import RolloutLogger
        method_name = cfg.eval.get("rollout_method_name", "cd-dp")
        task_name = cfg.eval.get("rollout_task_name", "cube-mv")
        rollout_logger = RolloutLogger(
            method=method_name,
            task=task_name,
            out_dir=results_path,
            n_envs=eval_world.n_eval,
            eval_budget=eval_world.eval_budget,
        )
        rollout_logger.set_episodes(
            episode_idx=eval_world.eval_episodes.tolist(),
            start_step=eval_world.eval_start_idx.tolist(),
            goal_offset=(eval_world.goal_offsets.tolist()
                         if hasattr(eval_world.goal_offsets, "tolist")
                         else [int(eval_world.goal_offsets)] * eval_world.n_eval),
        )
        rollout_logger.add_meta(
            policy=cfg.policy,
            eval_dataset=cfg.eval.dataset_name,
            train_dataset=cfg.dataset.get("stats", cfg.eval.dataset_name)
                if hasattr(cfg, "dataset") else cfg.eval.dataset_name,
            cache_dir=cfg.get("cache_dir", None),
            seed=int(cfg.seed),
            action_block=int(cfg.plan_config.action_block),
            horizon=int(cfg.plan_config.horizon),
            receding_horizon=int(cfg.plan_config.receding_horizon),
            plan_config=cfg.plan_config,
            world=cfg.world,
            diffusion_ckpt=str(diff_cfg.ckpt_path),
            dp_arch={
                "horizon": int(dp.horizon), "K": int(dp.K),
                "stride": int(dp.stride), "action_dim": int(dp.action_dim),
                "obs_steps": int(dp.obs_steps),
                "action_steps": int(dp.action_steps),
                "image_size": int(dp.image_size),
                "n_views": int(dp.n_views),
                "sample_steps": int(dp.sample_steps),
                "backbone": str(cfg.diffusion.get("nn", "chi_unet")),
            },
            n_actions_per_replan=int(n_actions_needed),
            start_offset=cfg.eval.get("start_offset", None),
        )
        rollout_logger.attach_demo_ttc(cfg.eval.get("demo_ttc_path", None))
        print(f"[rollout_log] enabled — writing to {results_path}")

    t0 = time.time()
    metrics = evaluate_tracking_from_dataset(
        world=eval_world.world,
        dataset=eval_world.dataset,
        solver=solver,
        episodes_idx=eval_world.eval_episodes.tolist(),
        start_steps=eval_world.eval_start_idx.tolist(),
        goal_offset_steps=(eval_world.goal_offsets.tolist()
                           if hasattr(eval_world.goal_offsets, "tolist")
                           else eval_world.goal_offsets),
        eval_budget=eval_world.eval_budget,
        callables=callables,
        video_path=str(results_path),
        save_video=cfg.eval.get("save_video", True),
        debug=False,
        model=None,
        logger=rollout_logger,
    )
    elapsed = time.time() - t0
    print(
        f"[cd-eval] done in {elapsed:.1f}s  "
        f"replans={policy._n_replans}  ring_updates={policy._n_ring_updates}"
    )

    # -- Phase 7: write text summary --
    txt = results_path / "cd_dp_results.txt"
    with open(txt, "w") as f:
        f.write(f"dp_ckpt        : {diff_cfg.ckpt_path}  (CleanDiffuser)\n")
        f.write(f"meta arch      : horizon={dp.horizon} K={dp.K} action_dim={dp.action_dim} "
                f"V={dp.n_views} img={dp.image_size} nn={cfg.diffusion.get('nn','chi_unet')}\n")
        f.write(f"actions/replan : {n_actions_needed} (available {dp.horizon})\n")
        f.write(f"sample         : {dp.sample_steps}/ddpm  w_cfg=1.0\n")
        f.write(f"start_offset   : {cfg.eval.get('start_offset', '?')}\n")
        f.write(f"replans        : {policy._n_replans}\n")
        f.write(f"ring_updates   : {policy._n_ring_updates}\n")
        if metrics:
            for k, v in metrics.items():
                f.write(f"{k:14s} : {v}\n")
        f.write(f"elapsed (s)    : {elapsed:.1f}\n")
    print(f"[cd-eval] summary -> {txt}")


if __name__ == "__main__":
    run()
