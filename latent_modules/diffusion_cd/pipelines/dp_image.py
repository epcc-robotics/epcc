"""Pixel-DP training — CleanDiffuser canonical recipe (task-agnostic entry).

Dataset class is selected by the Hydra config's `dataset._target_`. Pair with
a task-specific yaml under `configs/dp/<task>/<arch>/<task>_image.yaml`.

Submit examples (sbatch):
    # Cube AW (current best — 100% Full GTR):
    DATASET_PATH=$STABLEWM_HOME/datasets/cube_multiview_diffusion_c0.3_postpad_fullscale.h5 \
    sbatch submit_train_dp_cube_aw.sh
    # PushT SV: see submit_train_dp_pusht_sv.sh
"""
import os
import sys
import time
import pathlib
import warnings
warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# `utils.py` lives in CleanDiffuser's `pipelines/` dir; same path style as
# their canonical scripts.
from utils import set_seed, Logger

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters


@hydra.main(config_path="../configs/dp/cube_aw/chi_unet",
            config_name="cube_aw_image",
            version_base=None)
def pipeline(args):
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)

    # ---- Dataset (Hydra dispatch via dataset._target_) ----
    dataset = hydra.utils.instantiate(
        args.dataset,
        pad_before=args.obs_steps - 1,
        pad_after=args.action_steps - 1,
    )
    print(dataset)

    # ---- frame-level train/val split (95/5) for overfit detection ----
    # Train and val share episodes but disjoint window indices — not a true
    # OOD val, but a useful overfit proxy. Episode-level split would require
    # re-running Jpeg2k compression which is expensive.
    all_indices = np.arange(len(dataset))
    rng = np.random.default_rng(int(args.seed))
    rng.shuffle(all_indices)
    n_val_idx = max(1, int(round(len(all_indices) * 0.05)))
    val_indices = all_indices[:n_val_idx].tolist()
    train_indices = all_indices[n_val_idx:].tolist()
    print(f"[train/val split] {len(train_indices)} train windows, {len(val_indices)} val windows")

    from torch.utils.data import SubsetRandomSampler
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    dl_val = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # ---- Model (mirrors dp_robomimic_image.py) ----
    if args.nn == "chi_unet":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import ChiUNet1d
        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=256,
            rgb_model_name=args.rgb_model,
            resize_shape=args.resize_shape,
            crop_shape=args.crop_shape,
            random_crop=args.random_crop,
            use_group_norm=args.use_group_norm,
            use_seq=args.use_seq,
        ).to(args.device)
        nn_diffusion = ChiUNet1d(
            args.action_dim, 256, args.obs_steps,
            model_dim=256, emb_dim=256, dim_mult=[1, 2, 2],
            obs_as_global_cond=True,
            timestep_emb_type="positional",
        ).to(args.device)
    elif args.nn == "dit":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import DiT1d
        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=256,
            rgb_model_name=args.rgb_model,
            resize_shape=args.resize_shape,
            crop_shape=args.crop_shape,
            random_crop=args.random_crop,
            use_group_norm=args.use_group_norm,
            use_seq=args.use_seq,
        ).to(args.device)
        nn_diffusion = DiT1d(
            args.action_dim, emb_dim=256 * args.obs_steps,
            d_model=320, n_heads=10, depth=2,
            timestep_emb_type="fourier",
        ).to(args.device)
    elif args.nn == "chi_transformer":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import ChiTransformer
        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=256,
            rgb_model_name=args.rgb_model,
            resize_shape=args.resize_shape,
            crop_shape=args.crop_shape,
            random_crop=args.random_crop,
            use_group_norm=args.use_group_norm,
            use_seq=args.use_seq,
            keep_horizon_dims=True,
        ).to(args.device)
        nn_diffusion = ChiTransformer(
            args.action_dim, 256, args.horizon, args.obs_steps,
            d_model=256, nhead=4, num_layers=4,
            timestep_emb_type="positional",
        ).to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")

    print("======== Parameter Report (diffusion) ========")
    report_parameters(nn_diffusion)
    print("======== Parameter Report (condition) ========")
    report_parameters(nn_condition)
    print("==============================================")

    x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device) * +1.0
    x_min = torch.ones((1, args.horizon, args.action_dim), device=args.device) * -1.0

    if args.diffusion == "ddpm":
        from cleandiffuser.diffusion.ddpm import DDPM
        agent = DDPM(
            nn_diffusion=nn_diffusion, nn_condition=nn_condition,
            device=args.device, diffusion_steps=args.sample_steps,
            x_max=x_max, x_min=x_min,
            optim_params={"lr": args.lr},
        )
    elif args.diffusion == "edm":
        from cleandiffuser.diffusion.edm import EDM
        agent = EDM(
            nn_diffusion=nn_diffusion, nn_condition=nn_condition,
            device=args.device, optim_params={"lr": args.lr},
        )
    else:
        raise NotImplementedError(f"diffusion={args.diffusion}")

    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps)

    # ---- Dump action normalizer stats (sibling file for eval-time reuse) ----
    # eval_cleandiffuser_dp.py loads this first; falls back to train-H5 reload
    # only when missing. Saved early so even an interrupted train still has stats.
    action_norm = dataset.normalizer["action"]
    stats_path = pathlib.Path(args.work_dir) / "normalizer_stats.npz"
    np.savez(
        stats_path,
        action_min=np.asarray(action_norm.min, dtype=np.float32),
        action_max=np.asarray(action_norm.max, dtype=np.float32),
    )
    print(f"[dp] saved normalizer_stats → {stats_path}", flush=True)

    # ---- Train (no in-loop env eval; ckpts saved for external eval) ----
    n_gradient_step = 0
    diffusion_loss_list = []
    start_time = time.time()
    for batch in loop_dataloader(dataloader):
        nobs = batch["obs"]
        condition = {k: nobs[k][:, :args.obs_steps, ...].to(args.device)
                     for k in nobs}
        naction = batch["action"].to(args.device)

        loss = agent.update(naction, condition)["loss"]
        lr_scheduler.step()
        diffusion_loss_list.append(loss)

        if n_gradient_step % args.log_freq == 0:
            avg = float(np.mean(diffusion_loss_list))
            print(f"[step {n_gradient_step:>7d}] loss={avg:.5f}  "
                  f"lr={lr_scheduler.get_last_lr()[0]:.2e}  "
                  f"t={time.time()-start_time:.0f}s", flush=True)
            logger.log({"step": n_gradient_step, "avg_loss": avg,
                        "total_time": time.time() - start_time},
                       category="train")
            diffusion_loss_list = []

        # Val: random-t denoising MSE + full-DDPM-sample L2 vs GT.
        # Note: MultiImageObsCondition mutates obs_dict in place (flattens),
        # so we rebuild v_cond fresh between agent.loss and agent.sample.
        if n_gradient_step > 0 and n_gradient_step % int(args.get("val_freq", 2000)) == 0:
            vbatch = next(iter(dl_val))
            v_nobs = vbatch["obs"]
            v_action = vbatch["action"].to(args.device)
            with torch.no_grad():
                # (a) random-t denoising MSE — fresh cond
                v_cond_loss = {k: v_nobs[k][:, :args.obs_steps, ...].to(args.device)
                               for k in v_nobs}
                vloss = agent.loss(v_action, v_cond_loss).item()
                # (b) full DDPM sample → L2 vs GT — REBUILD fresh cond
                v_cond_sample = {k: v_nobs[k][:, :args.obs_steps, ...].to(args.device)
                                 for k in v_nobs}
                B = v_action.shape[0]
                prior = torch.zeros_like(v_action)
                sampled, _ = agent.sample(
                    prior, n_samples=B,
                    sample_steps=int(args.sample_steps),
                    use_ema=True, temperature=float(args.temperature),
                    condition_cfg=v_cond_sample, w_cfg=1.0,
                )
                vsample_l2 = ((sampled - v_action) ** 2).mean().item()
            print(f"  [val @ step {n_gradient_step}] loss={vloss:.5f}  sample_l2={vsample_l2:.5f}",
                  flush=True)
            logger.log({"step": n_gradient_step, "val_loss": vloss,
                        "val_sample_l2": vsample_l2}, category="train")

        if n_gradient_step > 0 and n_gradient_step % args.save_freq == 0:
            logger.save_agent(agent=agent, identifier=n_gradient_step)

        n_gradient_step += 1
        if n_gradient_step > args.gradient_steps:
            logger.save_agent(agent=agent, identifier=n_gradient_step)
            logger.finish(agent)
            break


if __name__ == "__main__":
    pipeline()
