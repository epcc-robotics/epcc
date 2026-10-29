"""Latent-trajectory diffusion training — CleanDiffuser + lvel (task-agnostic).

Dataset class is selected by the Hydra config's `dataset._target_`. Pair with
a task-specific yaml under `configs/<task>_latent/<ablation>.yaml`.

Trains one of six ablations (A/B/C × std/mm) governed by Hydra config.
Hydra branches:
  args.nn               in {chi_unet, dit}
  args.predict_noise    bool (False = x0, True = ε)
  args.normalizer_kind  in {standard, minmax}

Saves periodic ckpts + a `meta.npz` next to ckpts so eval can rebuild
the normalizer from the same stats.
"""
from __future__ import annotations

import os
import sys
import time
import pathlib
import json
import warnings
warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import set_seed, Logger

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from latent_diffusion_cd.diffusion.ddpm_with_vel import DDPMWithVelLoss
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.utils import report_parameters


def _build_backbone(args, latent_dim, history, horizon, emb_dim, device):
    if args.nn == "chi_unet":
        from cleandiffuser.nn_diffusion import ChiUNet1d
        # act_dim = latent dim (we're diffusing latents, not actions).
        return ChiUNet1d(
            latent_dim, emb_dim, history,
            model_dim=int(args.model_dim), emb_dim=emb_dim,
            dim_mult=list(args.get("dim_mult", [1, 2, 2])),
            obs_as_global_cond=True,
            timestep_emb_type="positional",
        ).to(device)
    if args.nn == "dit":
        from cleandiffuser.nn_diffusion import DiT1d
        return DiT1d(
            latent_dim, emb_dim=emb_dim * history,
            d_model=int(args.model_dim), n_heads=int(args.get("n_heads", 6)),
            depth=int(args.depth), timestep_emb_type="fourier",
        ).to(device)
    if args.nn == "chi_transformer":
        from cleandiffuser.nn_diffusion import ChiTransformer
        return ChiTransformer(
            latent_dim, emb_dim, horizon, history,
            d_model=int(args.model_dim), nhead=int(args.get("n_heads", 6)),
            num_layers=int(args.depth), timestep_emb_type="positional",
        ).to(device)
    raise ValueError(f"unknown args.nn={args.nn}")


@hydra.main(
    config_path="../configs/cube_latent",
    config_name="A_std",
    version_base=None,
)
def pipeline(args):
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)

    # ---- dataset (Hydra dispatch via dataset._target_) ----
    ds_train = hydra.utils.instantiate(args.dataset, split="train")
    ds_val = hydra.utils.instantiate(args.dataset, split="val")
    print(f"[train] H={args.horizon} K={args.history} stride={args.stride}  "
          f"latent_dim={ds_train.latent_dim}  "
          f"train_windows={len(ds_train)}  val_windows={len(ds_val)}")
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=int(args.batch_size), num_workers=int(args.num_workers),
        shuffle=True, pin_memory=True, persistent_workers=True, drop_last=True,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val, batch_size=int(args.batch_size), num_workers=int(args.num_workers),
        shuffle=False, pin_memory=True, persistent_workers=True, drop_last=False,
    )

    latent_dim = ds_train.latent_dim
    horizon = int(args.horizon)
    history = int(args.history)
    cond_emb_dim = int(args.cond_emb_dim)

    # ---- cond module (MLP over flat history) ----
    # CleanDiffuser convention: cond output is (B, K*emb_dim) flat — both
    # ChiUNet1d (internal flatten matches K*emb_dim Linear) and DiT1d
    # (emb_dim arg is K*emb_dim, added to noise embedding) expect this.
    total_cond_dim = history * cond_emb_dim
    nn_condition = MLPCondition(
        in_dim=history * latent_dim,
        out_dim=total_cond_dim,
        hidden_dims=[int(args.cond_hidden)],
        dropout=float(args.get("cond_dropout", 0.0)),
    ).to(args.device)

    # ---- backbone ----
    # cond_emb_dim is per-step; total flat dim seen by the backbone = K*emb_dim.
    nn_diffusion = _build_backbone(
        args, latent_dim=latent_dim,
        history=history, horizon=horizon,
        emb_dim=cond_emb_dim, device=args.device,
    )

    print("======== Parameter Report (diffusion) ========")
    report_parameters(nn_diffusion)
    print("======== Parameter Report (condition) ========")
    # Skip cleandiffuser.report_parameters for nn_condition: MLPCondition has
    # ~5 params (< default topk=10) → triggers IndexError. Print manual summary.
    cond_total = sum(p.numel() for p in nn_condition.parameters() if p.requires_grad)
    print(f"Total parameters: {cond_total:,}")
    print("==============================================")

    # ---- fix_mask: anchor first latent frame (inpainting-style) ----
    fix_mask = torch.zeros(horizon, latent_dim)
    if bool(args.get("fix_mask_first", True)):
        fix_mask[0, :] = 1.0
    print(f"[train] fix_mask: first-frame anchor = {bool(args.get('fix_mask_first', True))}")

    # ---- DDPMWithVelLoss agent ----
    # x_max / x_min clip — only when explicitly enabled in config.
    x_max = x_min = None
    if bool(args.get("x_clip", False)):
        # MinMax mode: clip to [-1, +1]; Standard mode: clip to ±x_clip_abs (typically 5).
        bound = float(args.get("x_clip_abs", 1.0))
        x_max = torch.ones((1, horizon, latent_dim), device=args.device) * +bound
        x_min = torch.ones((1, horizon, latent_dim), device=args.device) * -bound

    agent = DDPMWithVelLoss(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition,
        fix_mask=fix_mask, loss_weight=None,
        device=args.device,
        diffusion_steps=int(args.diffusion_steps),
        ema_rate=float(args.ema_rate),
        optim_params={"lr": float(args.lr)},
        x_max=x_max, x_min=x_min,
        predict_noise=bool(args.predict_noise),
        beta_schedule=str(args.get("beta_schedule", "cosine")),
        lambda_vel=float(args.lambda_vel),
        lambda_acc=float(args.lambda_acc),
        motion_only_pad=bool(args.get("motion_only_pad", False)),
    )

    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=int(args.gradient_steps))

    # ---- write a tiny meta file alongside ckpts (eval-time reuse) ----
    meta = {
        "nn": args.nn,
        "predict_noise": bool(args.predict_noise),
        "normalizer_kind": args.normalizer_kind,
        "lambda_vel": float(args.lambda_vel),
        "lambda_acc": float(args.lambda_acc),
        "horizon": horizon,
        "history": history,
        "stride": int(args.stride),
        "latent_dim": latent_dim,
        "model_dim": int(args.model_dim),
        "cond_emb_dim": cond_emb_dim,
        "cond_hidden": int(args.cond_hidden),
        "depth": int(args.get("depth", 0)),
        "dim_mult": list(args.get("dim_mult", [1, 2, 2])),
        "diffusion_steps": int(args.diffusion_steps),
        "fix_mask_first": bool(args.get("fix_mask_first", True)),
        "beta_schedule": str(args.get("beta_schedule", "cosine")),
        "h5_path": str(args.h5_path),
        "padding_mode": str(args.padding_mode),
    }
    work_dir = pathlib.Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    with open(work_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] wrote meta -> {work_dir / 'meta.json'}")

    # Save normalizer stats. MinMaxNormalizer has .min/.max (no .mean);
    # StandardScalerNormalizer has .mean/.std. Save what's available — eval-side
    # _reconstruct_normalizer picks the right keys based on meta["normalizer_kind"].
    norm = ds_train.normalizer
    np.savez(work_dir / "normalizer_stats.npz",
             kind=args.normalizer_kind,
             mean=np.asarray(getattr(norm, "mean", np.zeros(latent_dim)), dtype=np.float32),
             std=np.asarray(getattr(norm, "std", np.ones(latent_dim)), dtype=np.float32),
             min=np.asarray(getattr(norm, "min", np.zeros(latent_dim)), dtype=np.float32),
             max=np.asarray(getattr(norm, "max", np.ones(latent_dim)), dtype=np.float32))
    print(f"[train] wrote normalizer_stats -> {work_dir / 'normalizer_stats.npz'}")

    # ---- training loop ----
    n_step = 0
    loss_buf = []
    t0 = time.time()
    for batch in loop_dataloader(dl_train):
        obs = batch["obs"].to(args.device)            # (B, H, D)
        cond = batch["cond"].to(args.device)          # (B, K*D)
        is_pad = batch["is_pad"].to(args.device) if bool(args.get("use_is_pad", False)) else None

        out = agent.update(obs, condition=cond, is_pad=is_pad)
        lr_scheduler.step()
        loss_buf.append(out["loss"])

        if n_step % int(args.log_freq) == 0:
            avg = float(np.mean(loss_buf))
            lr_now = lr_scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            print(f"[step {n_step:>7d}] loss={avg:.5f}  lr={lr_now:.2e}  t={elapsed:.0f}s",
                  flush=True)
            logger.log({"step": n_step, "avg_loss": avg,
                        "total_time": elapsed}, category="train")
            loss_buf = []

        if n_step > 0 and n_step % int(args.val_freq) == 0 and len(ds_val) > 0:
            # Val: 1 batch — measure both (a) random-t denoising MSE and
            # (b) full-sample L2 to GT (deploy-relevant overfit detector).
            vbatch = next(iter(dl_val))
            with torch.no_grad():
                vobs = vbatch["obs"].to(args.device)
                vcond = vbatch["cond"].to(args.device)
                vpad = vbatch["is_pad"].to(args.device) if is_pad is not None else None
                # (a) random-t denoising MSE (same as train loss, on val)
                vloss = agent.loss(vobs, condition=vcond, is_pad=vpad).item()
                # (b) full DDPM sample → x0 → L2 vs GT (50 forward passes, ~2-3s)
                B = vobs.shape[0]
                prior = torch.zeros_like(vobs)
                sampled, _ = agent.sample(
                    prior, n_samples=B,
                    sample_steps=int(args.diffusion_steps),
                    use_ema=True, temperature=1.0,
                    condition_cfg=vcond, w_cfg=1.0,
                )
                vsample_l2 = ((sampled - vobs) ** 2).mean().item()
            print(f"  [val @ step {n_step}] loss={vloss:.5f}  sample_l2={vsample_l2:.5f}",
                  flush=True)
            logger.log({"step": n_step, "val_loss": vloss,
                        "val_sample_l2": vsample_l2}, category="train")

        if n_step > 0 and n_step % int(args.save_freq) == 0:
            logger.save_agent(agent=agent, identifier=n_step)

        n_step += 1
        if n_step > int(args.gradient_steps):
            logger.save_agent(agent=agent, identifier=n_step)
            logger.finish(agent)
            break


if __name__ == "__main__":
    pipeline()
