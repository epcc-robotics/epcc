"""Diffusion-MPC planner module — builds the trained diffusion planner from
ckpt + sidecar stats and generates per-env latent references for cost.py.

The planner is the velocity-loss diffusion SDE from `train_cube_velloss.py`.
This module is the single source of truth for diffusion-MPC inference; both
`eval_diffusion_mpc.py` (rewrite) and any future ablation scripts should call
into here so the load/sample/expand contract stays consistent.

Three pieces:
    1. `build_diffusion_planner(ckpt_path, device, **overrides)` — loads ckpt,
       auto-reads arch from `meta` dict (set by `_save_with_meta` in train),
       loads sidecar `stats.npz` for latent normalization. Returns a
       `DiffusionPlanner` dataclass.
    2. `diffusion_generate_latent_ref(diff, history_embs, action_block,
       **overrides)` — Janner inpaint (`prior[:,0] = hist_norm[-1]`), sample
       via cleandiffuser, denormalize, `repeat_interleave(action_block)` so
       cost.py's stride-5 indexing hits each unique frame.
    3. `LatentScaler` — light wrapper for normalize / denormalize, exposes
       mean/std as buffers on the active device.

Design notes:
    - Backward-incompatible with E_v1..E_v4 ckpts that do NOT have a `meta`
      dict — those need explicit `obs_dim/horizon/...` overrides. Sanity_c1_H6
      and onward auto-load.
    - `x_clip <= 0` means NO clipping (matches train CLI semantics; the
      tangled backup had a bug treating 0 as ±0 → forced 0 vector).
    - History conditioning is HISTORY-ONLY (no goal token) per
      train_cube_velloss.py:116. Diffusion is implicitly goal-conditioned
      via the fixed-goal training distribution.
    - `predict_x0` is FORCED True by VelLossDiffusionSDE.__init__ regardless
      of cfg flag, so we read it from meta but assert consistency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d, JannerUNet1d

from .diffusion_with_vel_loss import VelLossDiffusionSDE


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class DiffusionPlanner:
    """Bundle of everything needed to sample latent references at runtime."""
    planner: Any                      # cleandiffuser VelLossDiffusionSDE
    scaler: "LatentScaler"            # mean/std for normalize/denormalize
    meta: dict                        # arch + sample-time flags from ckpt
    device: torch.device
    K: int                            # history length
    H: int                            # horizon (= ref length before expand)
    stride: int                       # diffusion-time stride between frames
    obs_dim: int                      # latent dim (e.g. 384 AW)
    predict_x0: bool                  # always True for VelLoss ckpts
    sample_steps: int                 # default DDIM step count
    solver: str                       # default ODE solver
    w_cfg: float                      # CFG weight (1.0 = no CFG)
    temperature: float                # noise temperature
    x_clip_abs: float | None          # None = no clip; > 0 = clamp


class LatentScaler:
    """Per-dim normalize / denormalize against `stats.npz` mean/std."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        # both (D,)
        self.mean = mean
        self.std = std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def to(self, device) -> "LatentScaler":
        return LatentScaler(self.mean.to(device), self.std.to(device))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _load_meta(ckpt_path: str) -> tuple[dict, dict]:
    """Read ckpt, return (state, meta). meta is empty dict for legacy ckpts."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        return state, {}
    return state, dict(state.get("meta", {}))


def _build_nn_diffusion(meta: dict):
    """Reconstruct backbone matching training-time `build_planner`."""
    if meta["planner_net"] == "unet":
        return JannerUNet1d(
            meta["obs_dim"],
            model_dim=meta.get("unet_dim", 64),
            emb_dim=meta.get("unet_dim", 64),
            timestep_emb_type="positional",
            attention=False,
            kernel_size=5,
        )
    if meta.get("cross_attn_dit", False):
        from .dit_crossattn import DiT1dCrossAttn
        return DiT1dCrossAttn(
            meta["obs_dim"],
            emb_dim=meta["planner_emb"],
            d_model=meta["d_model"],
            n_heads=meta["d_model"] // 64,
            depth=meta["depth"],
            timestep_emb_type="fourier",
            history_dim=meta["obs_dim"],
            K=meta["history"],
            cross_gate=meta.get("cross_gate", False),
        )
    return DiT1d(
        meta["obs_dim"],
        emb_dim=meta["planner_emb"],
        d_model=meta["d_model"],
        n_heads=meta["d_model"] // 64,
        depth=meta["depth"],
        timestep_emb_type="fourier",
    )


def build_diffusion_planner(
    ckpt_path: str,
    device: str | torch.device = "cuda",
    *,
    overrides: dict | None = None,
) -> DiffusionPlanner:
    """Reconstruct + load the trained diffusion planner.

    Args:
        ckpt_path: path to a `.pt` saved by `_save_with_meta` (post-2026-04-26)
            or a legacy ckpt — legacy requires `overrides` with arch fields.
        device: where to put the planner (cuda recommended).
        overrides: optional dict of meta fields to override (or supply if the
            ckpt has no `meta`). Useful for legacy E_v1..E_v4 ckpts and for
            inference-time tweaks (`sample_steps`, `w_cfg`, `x_clip_abs`).

    Returns:
        DiffusionPlanner containing the loaded planner + scaler + resolved meta.
    """
    device = torch.device(device)
    state, meta = _load_meta(ckpt_path)
    if overrides:
        meta = {**meta, **overrides}

    required = ("obs_dim", "horizon", "history", "stride", "planner_net",
                "depth", "d_model", "planner_emb", "cond_hidden", "cfg_dropout")
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(
            f"ckpt {ckpt_path} missing meta fields {missing}. Pass via "
            "`overrides=` for legacy ckpts."
        )

    # Auto-detect cross-attn from ckpt state if meta flag absent (back-compat
    # for ckpts trained before meta started recording the flag).
    if not meta.get("cross_attn_dit", False):
        sd = state.get("model", state)
        if isinstance(sd, dict) and any(k.startswith("diffusion.history_proj")
                                        for k in sd.keys()):
            meta["cross_attn_dit"] = True
    # Auto-detect cross_gate from adaLN_modulation linear weight: 6-chunk
    # weight has shape (6 * d_model, d_model); 7-chunk has (7 * d_model, d_model).
    if meta.get("cross_attn_dit", False) and "cross_gate" not in meta:
        sd = state.get("model", state)
        if isinstance(sd, dict):
            w_key = "diffusion.blocks.0.adaLN_modulation.1.weight"
            if w_key in sd:
                out_dim = sd[w_key].shape[0]
                meta["cross_gate"] = (out_dim == 7 * meta["d_model"])

    nn_diffusion = _build_nn_diffusion(meta)
    cond_out_dim = (meta.get("unet_dim", 64) if meta["planner_net"] == "unet"
                    else meta["planner_emb"])
    if meta.get("cross_attn_dit", False):
        from cleandiffuser.nn_condition import IdentityCondition
        nn_condition = IdentityCondition(dropout=meta.get("cfg_dropout", 0.0))
    else:
        nn_condition = MLPCondition(
            in_dim=meta["history"] * meta["obs_dim"],
            out_dim=cond_out_dim,
            hidden_dims=[meta["cond_hidden"]],
            act=nn.SiLU(),
            dropout=meta.get("cfg_dropout", 0.0),
        )

    fix_mask = torch.zeros((meta["horizon"], meta["obs_dim"]))
    fix_mask[0, :] = 1.0

    x_clip_abs = meta.get("x_clip_abs", None)
    x_max = x_min = None
    if x_clip_abs is not None and float(x_clip_abs) > 0:
        x_max = torch.full((1, 1, meta["obs_dim"]), float(x_clip_abs))
        x_min = torch.full((1, 1, meta["obs_dim"]), -float(x_clip_abs))

    planner = VelLossDiffusionSDE(
        nn_diffusion,
        nn_condition=nn_condition,
        fix_mask=fix_mask,
        loss_weight=torch.ones((meta["horizon"], meta["obs_dim"])),
        grad_clip_norm=1.0,
        ema_rate=meta.get("ema_rate", 0.9999),
        optim_params={"lr": 2e-4, "weight_decay": 1e-5},
        device=device,
        noise_schedule="linear",
        x_max=x_max,
        x_min=x_min,
        lambda_vel=meta.get("lambda_vel", 0.0),
        lambda_acc=0.0,
    )

    # cleandiffuser's load() expects checkpoint["model"]; flat ckpts after the
    # 2026-04-26 _save_with_meta fix have that. For nested/legacy variants,
    # rewrite a temp dict on the fly.
    if "model" not in state and "state_dict" in state and "model" in state["state_dict"]:
        # legacy nested wrap — repackage transiently
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
            torch.save(state["state_dict"], tf.name)
            planner.load(tf.name)
            os.unlink(tf.name)
    else:
        planner.load(ckpt_path)
    planner.eval()

    # Sidecar scaler stats
    stats_path = meta.get("stats_path", "stats.npz")
    if not os.path.isabs(stats_path):
        stats_path = os.path.join(os.path.dirname(ckpt_path), stats_path)
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"stats.npz not found at {stats_path}. Required for normalization. "
            "Pass via `overrides={'stats_path': '/abs/path'}` if relocated."
        )
    stats = np.load(stats_path)
    mean = torch.from_numpy(stats["mean"]).float().to(device)
    std = torch.from_numpy(stats["std"]).float().to(device)
    scaler = LatentScaler(mean, std)

    print(f"[diff-planner] loaded {Path(ckpt_path).name}  "
          f"obs={meta['obs_dim']} K={meta['history']} H={meta['horizon']} "
          f"S={meta['stride']} depth={meta['depth']} "
          f"x_clip={'OFF' if (x_max is None) else float(x_clip_abs)}  "
          f"sample={meta.get('sample_steps', 32)}/{meta.get('planner_solver', 'ddim')}")

    return DiffusionPlanner(
        planner=planner,
        scaler=scaler,
        meta=meta,
        device=device,
        K=int(meta["history"]),
        H=int(meta["horizon"]),
        stride=int(meta["stride"]),
        obs_dim=int(meta["obs_dim"]),
        predict_x0=bool(meta.get("predict_x0", True)),
        sample_steps=int(meta.get("sample_steps", 32)),
        solver=str(meta.get("planner_solver", "ddim")),
        w_cfg=float(meta.get("w_cfg", 1.0)),
        temperature=float(meta.get("temperature", 1.0)),
        x_clip_abs=(float(x_clip_abs) if (x_clip_abs is not None and float(x_clip_abs) > 0) else None),
    )


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

def diffusion_generate_latent_ref(
    diff: DiffusionPlanner,
    history_embs: torch.Tensor,
    action_block: int,
    *,
    sample_steps: int | None = None,
    solver: str | None = None,
    w_cfg: float | None = None,
    temperature: float | None = None,
    use_first_n: int | None = None,
) -> torch.Tensor:
    """Sample one diffusion trajectory and expand to env-step granularity.

    Args:
        diff: DiffusionPlanner from `build_diffusion_planner`.
        history_embs: (K, D) raw (un-normalized) latent history. K must equal
            diff.K. D must equal diff.obs_dim.
        action_block: cost.py's stride for ref indexing (typically 5). Each
            diffusion frame is repeat_interleaved this many times so cost.py's
            t_offsets `[0, ab, 2*ab, ...]` hit each unique frame exactly once.
        sample_steps / solver / w_cfg / temperature: optional overrides; default
            to diff.* (= training-time eval defaults).

    Returns:
        ref_traj: (H * action_block, D) raw (denormalized) latent tensor on
            diff.device. For sanity_c1_H6 (H=6, action_block=5): (30, D).
    """
    K, D = history_embs.shape
    assert K == diff.K, f"history K={K} != trained K={diff.K}"
    assert D == diff.obs_dim, f"history D={D} != trained D={diff.obs_dim}"
    # cost.py indexes ref at stride = WM action_block. Training stride must
    # match runtime action_block, otherwise repeat_interleave produces ref
    # frames at the wrong cadence (e.g. stride=5 trained but ab=1 runtime
    # would 5×-stretch every output frame — silent CFG/cond mismatch).
    assert action_block == diff.stride, (
        f"action_block ({action_block}) != diff.stride ({diff.stride}). "
        "Diffusion was trained at stride=diff.stride; cost.py indexes ref at "
        "action_block. Mismatch produces duplicated or skipped ref frames."
    )

    history_embs = history_embs.to(diff.device)
    hist_norm = diff.scaler.normalize(history_embs)
    cond = hist_norm.reshape(1, -1)                    # (1, K*D)

    # Janner inpaint: fix output[0] = last history frame (anchor)
    prior = torch.zeros(1, diff.H, D, device=diff.device)
    prior[:, 0, :] = hist_norm[-1]

    with torch.no_grad():
        traj_norm, _ = diff.planner.sample(
            prior,
            solver=(solver if solver is not None else diff.solver),
            n_samples=1,
            sample_steps=(sample_steps if sample_steps is not None else diff.sample_steps),
            use_ema=True,
            condition_cfg=cond,
            w_cfg=(w_cfg if w_cfg is not None else diff.w_cfg),
            temperature=(temperature if temperature is not None else diff.temperature),
        )

    traj_raw = diff.scaler.denormalize(traj_norm.squeeze(0))   # (H, D)
    if use_first_n is not None and use_first_n < traj_raw.shape[0]:
        # Use only first N entries (e.g. H=18 ckpt + use_first_n=6 for MPC at
        # cost.py's T=6 horizon). The diffusion model was conditioned to
        # produce H frames coherently; the first N should be the most accurate.
        traj_raw = traj_raw[:int(use_first_n)]
    if action_block > 1:
        # cost.py reads ref[ref_step + t * action_block], t in [0, T-1]; each
        # diffusion frame must span action_block ref slots so the indexing
        # picks unique frames at [0, +ab, +2*ab, ...].
        traj_raw = traj_raw.repeat_interleave(action_block, dim=0)
    return traj_raw.contiguous()


def diffusion_generate_latent_ref_batched(
    diff: DiffusionPlanner,
    history_embs_batch: torch.Tensor,
    action_block: int,
    *,
    sample_steps: int | None = None,
    solver: str | None = None,
    w_cfg: float | None = None,
    temperature: float | None = None,
    use_first_n: int | None = None,
    expand: bool = True,
) -> torch.Tensor:
    """Batched variant — sample B diffusion trajectories in ONE call.

    Used by ensemble diffusion-MPC to sample N candidate refs per env in
    a single forward pass. cleandiffuser's `planner.sample` natively
    supports `n_samples=B` with `prior` and `cond` batched on dim 0; this
    helper mirrors `diffusion_generate_latent_ref` but vectorized.

    Args:
        history_embs_batch: (B, K, D) raw latent histories. K, D must match
            `diff.K`, `diff.obs_dim`. B is the effective batch (e.g. n_envs * N).
        action_block: same as `diffusion_generate_latent_ref`. Only used /
            enforced when ``expand=True``.
        sample_steps / solver / w_cfg / temperature / use_first_n: same.
        expand: When True (default, MPC mode), enforce
            ``action_block == diff.stride`` and ``repeat_interleave(action_block)``
            so cost.py's stride-action_block lookup hits each unique frame
            once. When False (MPCC mode), skip both: the caller (e.g.
            ``DiffusionMPCCPolicy._refs_from_ring``) wants raw fine-grained
            waypoints (B, H, D) for arc-length parametrization in
            ReferenceTrajectory — repeat_interleave would create duplicate
            adjacent waypoints with zero segment length, breaking θ.

    Returns:
        - ``expand=True`` (default): (B, H_out * action_block, D) raw latent
          tensor. For sanity_c1_H6 (H=6, ab=5) with B=20: (20, 30, D).
        - ``expand=False``: (B, H_out, D) raw latent tensor. For
          mpcc_c0.3_H18 (H=18) with B=20: (20, 18, D).
    """
    if history_embs_batch.dim() == 2:
        # Allow (K, D) for B=1 convenience
        history_embs_batch = history_embs_batch.unsqueeze(0)
    B, K, D = history_embs_batch.shape
    assert K == diff.K, f"history K={K} != trained K={diff.K}"
    assert D == diff.obs_dim, f"history D={D} != trained D={diff.obs_dim}"
    if expand:
        assert action_block == diff.stride, (
            f"action_block ({action_block}) != diff.stride ({diff.stride}). "
            "Mismatch produces duplicated or skipped ref frames. "
            "(For MPCC arc-length mode, pass expand=False to skip this check.)"
        )

    history_embs_batch = history_embs_batch.to(diff.device)
    hist_norm = diff.scaler.normalize(history_embs_batch)             # (B, K, D)
    cond = hist_norm.reshape(B, -1)                                   # (B, K*D)

    # Janner inpaint: fix output[0] = last history frame, per row.
    prior = torch.zeros(B, diff.H, D, device=diff.device)
    prior[:, 0, :] = hist_norm[:, -1]

    with torch.no_grad():
        traj_norm, _ = diff.planner.sample(
            prior,
            solver=(solver if solver is not None else diff.solver),
            n_samples=B,
            sample_steps=(sample_steps if sample_steps is not None else diff.sample_steps),
            use_ema=True,
            condition_cfg=cond,
            w_cfg=(w_cfg if w_cfg is not None else diff.w_cfg),
            temperature=(temperature if temperature is not None else diff.temperature),
        )

    traj_raw = diff.scaler.denormalize(traj_norm)                     # (B, H, D)
    if use_first_n is not None and use_first_n < traj_raw.shape[1]:
        traj_raw = traj_raw[:, :int(use_first_n), :]
    if expand and action_block > 1:
        traj_raw = traj_raw.repeat_interleave(action_block, dim=1)    # (B, H_out*ab, D)
    return traj_raw.contiguous()
