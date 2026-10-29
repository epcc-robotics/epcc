"""CleanDiffuser-trained latent diffusion ckpt adapter for eval_diffusion_mpcc.

Provides the bundle + builder + batched-sample fn needed to plug a
CleanDiffuser ckpt (saved by `train_cube_latent_velloss.py`) into the
existing DiffusionMPCCPolicy infrastructure.

Ckpt layout expected (saved by our CleanDiffuser train pipeline):
    <work_dir>/models/model_<step>.pt    — agent.save() dict {model, model_ema}
    <work_dir>/meta.json                  — arch + hparam meta (nn, predict_noise,
                                            normalizer_kind, lvel, horizon, etc.)
    <work_dir>/normalizer_stats.npz       — saved mean/std/min/max from training

NOTE: This module imports `cleandiffuser.*` lazily so the rest of
le-wm doesn't depend on the CleanDiffuser checkout being on PYTHONPATH
unless we're actually loading a CleanDiffuser ckpt.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


# =============================================================================
# Detection (used by eval_diffusion_mpcc.py to auto-dispatch)
# =============================================================================

def is_cleandiffuser_ckpt(ckpt_path: str) -> bool:
    """Return True if ckpt_path looks like a CleanDiffuser-format ckpt.

    Heuristic: sibling `meta.json` lives at <ckpt_dir>/../meta.json (because
    our train pipeline writes ckpts under `<work_dir>/models/model_<step>.pt`
    and meta at `<work_dir>/meta.json`).
    """
    ckpt_path = str(ckpt_path)
    if not os.path.isfile(ckpt_path):
        return False
    work_dir = Path(ckpt_path).parent.parent
    return (work_dir / "meta.json").is_file()


# =============================================================================
# Planner bundle (duck-typed to fit DiffusionMPCCPolicy via subclass)
# =============================================================================

@dataclass
class CleanDiffuserDiffPlanner:
    agent: Any                  # DDPMWithVelLoss instance (model + model_ema loaded)
    normalizer: Any             # StandardScaler / MinMax / Empty wrapper
    meta: dict                  # full meta.json contents
    device: torch.device
    # arch hyperparams
    K: int                      # history length
    H: int                      # latent traj length (horizon)
    stride: int                 # env-frames between adjacent latent frames
    obs_dim: int                # latent dim (e.g. 384)
    # inference hyperparams
    sample_steps: int
    w_cfg: float
    temperature: float
    predict_noise: bool
    normalizer_kind: str
    backbone: str               # "chi_unet" | "dit" | "chi_transformer"
    # surrogate fields to mimic DiffusionPlanner duck-type for logging
    x_clip_abs: float | None = None
    solver: str = "ddpm"


# =============================================================================
# Builder
# =============================================================================

def _reconstruct_normalizer(stats_path: str, kind: str):
    """Rebuild normalizer from saved stats.npz."""
    from latent_diffusion_cd.dataset.normalizers import StandardScalerNormalizer
    try:
        from cleandiffuser.dataset.dataset_utils import MinMaxNormalizer, EmptyNormalizer
    except ImportError:
        MinMaxNormalizer = EmptyNormalizer = None  # type: ignore

    stats = np.load(stats_path)
    if kind == "standard":
        norm = StandardScalerNormalizer.__new__(StandardScalerNormalizer)
        norm.mean = np.asarray(stats["mean"], dtype=np.float32)
        norm.std = np.asarray(stats["std"], dtype=np.float32)
        return norm
    if kind == "minmax":
        if MinMaxNormalizer is None:
            raise ImportError("MinMaxNormalizer not available in CleanDiffuser")
        norm = MinMaxNormalizer.__new__(MinMaxNormalizer)
        norm.min = np.asarray(stats["min"], dtype=np.float32)
        norm.max = np.asarray(stats["max"], dtype=np.float32)
        norm.range = (norm.max - norm.min).astype(np.float32)
        norm.range[norm.range < 1e-6] = 1.0
        return norm
    if kind == "none":
        if EmptyNormalizer is None:
            raise ImportError("EmptyNormalizer not available in CleanDiffuser")
        return EmptyNormalizer()
    raise ValueError(f"unknown normalizer kind {kind!r}")


def build_cleandiffuser_diff_planner(
    ckpt_path: str,
    device: str = "cuda",
) -> CleanDiffuserDiffPlanner:
    """Reconstruct DDPMWithVelLoss + load weights from a CleanDiffuser ckpt.

    Reads sibling files:
        <ckpt_dir>/../meta.json
        <ckpt_dir>/../normalizer_stats.npz
    """
    from latent_diffusion_cd.diffusion.ddpm_with_vel import DDPMWithVelLoss
    from cleandiffuser.nn_condition import MLPCondition
    from cleandiffuser.nn_diffusion import ChiUNet1d, DiT1d, ChiTransformer

    ckpt_path = str(ckpt_path)
    assert os.path.isfile(ckpt_path), f"ckpt not found: {ckpt_path}"
    work_dir = Path(ckpt_path).parent.parent
    meta_path = work_dir / "meta.json"
    stats_path = work_dir / "normalizer_stats.npz"
    assert meta_path.is_file(), f"meta.json missing at {meta_path}"
    assert stats_path.is_file(), f"normalizer_stats.npz missing at {stats_path}"

    with open(meta_path) as f:
        meta = json.load(f)
    print(f"[cd-planner] meta loaded from {meta_path}")
    print(f"  backbone={meta['nn']}  predict_noise={meta['predict_noise']}  "
          f"normalizer={meta['normalizer_kind']}  lvel={meta['lambda_vel']}")

    latent_dim = int(meta["latent_dim"])
    K = int(meta["history"])
    H = int(meta["horizon"])
    stride = int(meta["stride"])
    cond_emb_dim = int(meta["cond_emb_dim"])
    cond_hidden = int(meta["cond_hidden"])
    diffusion_steps = int(meta["diffusion_steps"])
    nn_kind = str(meta["nn"])
    predict_noise = bool(meta["predict_noise"])
    lambda_vel = float(meta["lambda_vel"])
    lambda_acc = float(meta.get("lambda_acc", 0.0))
    fix_mask_first = bool(meta.get("fix_mask_first", True))

    # nn_condition
    nn_condition = MLPCondition(
        in_dim=K * latent_dim,
        out_dim=K * cond_emb_dim,
        hidden_dims=[cond_hidden],
        dropout=0.0,
    ).to(device)

    # nn_diffusion (matches train's _build_backbone)
    if nn_kind == "chi_unet":
        nn_diffusion = ChiUNet1d(
            latent_dim, cond_emb_dim, K,
            model_dim=int(meta["model_dim"]), emb_dim=cond_emb_dim,
            dim_mult=list(meta.get("dim_mult", [1, 2, 2])),
            obs_as_global_cond=True,
            timestep_emb_type="positional",
        ).to(device)
    elif nn_kind == "dit":
        nn_diffusion = DiT1d(
            latent_dim, emb_dim=cond_emb_dim * K,
            d_model=int(meta["model_dim"]),
            n_heads=int(meta.get("n_heads", 6)),
            depth=int(meta["depth"]),
            timestep_emb_type="fourier",
        ).to(device)
    elif nn_kind == "chi_transformer":
        nn_diffusion = ChiTransformer(
            latent_dim, cond_emb_dim, H, K,
            d_model=int(meta["model_dim"]),
            nhead=int(meta.get("n_heads", 6)),
            num_layers=int(meta["depth"]),
            timestep_emb_type="positional",
        ).to(device)
    else:
        raise ValueError(f"unknown nn kind in meta: {nn_kind!r}")

    # fix_mask anchor
    fix_mask = torch.zeros(H, latent_dim)
    if fix_mask_first:
        fix_mask[0, :] = 1.0

    # agent (no x_clip at inference by default — caller may override later)
    agent = DDPMWithVelLoss(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition,
        fix_mask=fix_mask, loss_weight=None,
        device=device,
        diffusion_steps=diffusion_steps,
        ema_rate=0.9999,
        x_max=None, x_min=None,
        predict_noise=predict_noise,
        beta_schedule=str(meta.get("beta_schedule", "cosine")),
        lambda_vel=lambda_vel,
        lambda_acc=lambda_acc,
    )
    print(f"[cd-planner] loading agent from {ckpt_path}")
    agent.load(ckpt_path)
    agent.model.eval()
    agent.model_ema.eval()

    normalizer = _reconstruct_normalizer(str(stats_path), meta["normalizer_kind"])

    return CleanDiffuserDiffPlanner(
        agent=agent, normalizer=normalizer, meta=meta, device=device,
        K=K, H=H, stride=stride, obs_dim=latent_dim,
        sample_steps=diffusion_steps,
        w_cfg=1.0, temperature=1.0,
        predict_noise=predict_noise,
        normalizer_kind=str(meta["normalizer_kind"]),
        backbone=nn_kind,
        x_clip_abs=None,
        solver="ddpm",
    )


# =============================================================================
# Batched sample fn — mirrors diffusion_generate_latent_ref_batched signature
# =============================================================================

@torch.no_grad()
def cleandiffuser_generate_latent_ref_batched(
    planner: CleanDiffuserDiffPlanner,
    history_embs_batch: torch.Tensor,
    action_block: int,
    *,
    sample_steps: int | None = None,
    w_cfg: float | None = None,
    temperature: float | None = None,
    use_first_n: int | None = None,
    expand: bool = True,
) -> torch.Tensor:
    """B histories → B latent trajectories. Mirrors latent_diffusion.planner's fn."""
    if history_embs_batch.dim() == 2:
        history_embs_batch = history_embs_batch.unsqueeze(0)
    B, K, D = history_embs_batch.shape
    assert K == planner.K, f"K={K} != trained K={planner.K}"
    assert D == planner.obs_dim, f"D={D} != trained D={planner.obs_dim}"
    if expand:
        assert action_block == planner.stride, (
            f"action_block ({action_block}) != stride ({planner.stride})"
        )

    history_embs_batch = history_embs_batch.to(planner.device)

    # Normalize via the pluggable normalizer (numpy-side; same path as training).
    hist_np = history_embs_batch.detach().cpu().numpy().reshape(-1, D)
    hist_norm_np = planner.normalizer.normalize(hist_np).reshape(B, K, D)
    hist_norm = torch.from_numpy(hist_norm_np).float().to(planner.device)

    # Cond: (B, K*D) flat — matches training's MLPCondition in_dim.
    cond = hist_norm.reshape(B, -1)

    # Anchor: fix output[0] = last history frame (matches training fix_mask).
    prior = torch.zeros(B, planner.H, D, device=planner.device)
    prior[:, 0, :] = hist_norm[:, -1]

    traj_norm, _ = planner.agent.sample(
        prior, n_samples=B,
        sample_steps=int(sample_steps if sample_steps is not None else planner.sample_steps),
        use_ema=True,
        temperature=float(temperature if temperature is not None else planner.temperature),
        condition_cfg=cond,
        w_cfg=float(w_cfg if w_cfg is not None else planner.w_cfg),
    )

    # Denormalize.
    traj_np = traj_norm.detach().cpu().numpy().reshape(-1, D)
    traj_raw_np = planner.normalizer.unnormalize(traj_np).reshape(B, planner.H, D)
    traj_raw = torch.from_numpy(traj_raw_np).float().to(planner.device)

    if use_first_n is not None and use_first_n < traj_raw.shape[1]:
        traj_raw = traj_raw[:, :int(use_first_n), :]
    if expand and action_block > 1:
        traj_raw = traj_raw.repeat_interleave(action_block, dim=1)
    return traj_raw.contiguous()
