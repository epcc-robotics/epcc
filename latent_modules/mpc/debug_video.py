"""Debug video renderer for MPC evaluation.

Renders per-step composite debug frames showing:
- Agent rollout and expert reference side-by-side
- Cost trajectory plots (total, tracking, terminal, action)
- Three-row comparison: GT frames, decoded ref embeddings, decoded pred embeddings
- 2D latent space PCA projection with ref path + predicted trajectory

Usage:
    from latent_modules.mpc.debug_video import DebugLogger, render_debug_video
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import List

import imageio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


class DebugLogger:
    """Accumulates per-step debug data during MPC evaluation."""

    def __init__(self, n_envs: int, eval_budget: int):
        self.n_envs = n_envs
        self.eval_budget = eval_budget
        self.tracking_costs: List[np.ndarray] = []   # list of (n_envs,) arrays
        self.terminal_costs: List[np.ndarray] = []   # list of (n_envs,) arrays
        self.action_costs: List[np.ndarray] = []     # list of (n_envs,) arrays
        self.total_costs: List[np.ndarray] = []      # list of (n_envs,) arrays
        self.pred_embeddings: List[torch.Tensor] = []  # list of (n_envs, T, D)
        self.ref_embeddings: List[torch.Tensor] = []   # list of (T, D)
        self.ref_frame_indices: List[np.ndarray] = []  # list of (T,) int — indices into reference trajectory
        self.ref_steps: List[np.ndarray] = []          # list of (n_envs,) int

    def log_step(self, t: int, solver) -> None:
        """Call after world.step() and solver.advance_steps().

        Uses per-env breakdowns (keyed by ref_idx) so each env gets its own
        cost terms and predicted embeddings — no cross-trajectory mixing.
        """
        env_bds = solver.get_env_breakdowns()  # dict[ref_idx, breakdown]
        if not env_bds:
            return

        # Build per-env arrays from the per-ref_idx breakdowns
        tracking = np.zeros(self.n_envs)
        terminal = np.zeros(self.n_envs)
        action = np.zeros(self.n_envs)
        pred_list = []  # will become (n_envs, T, D)
        ref_list = []   # per-env z_ref: list of (T, D)

        for env_i in range(self.n_envs):
            ref_idx = int(solver._env_ref_indices[env_i])
            bd = env_bds.get(ref_idx)
            if bd is None:
                # Fallback: use any available breakdown
                bd = next(iter(env_bds.values()))

            # Costs: (B, S) -> mean over both dims -> scalar
            tracking[env_i] = bd["tracking"].mean().cpu().item()
            terminal[env_i] = bd["terminal"].mean().cpu().item()
            action[env_i] = bd["action"].mean().cpu().item()

            # Predicted embeddings: (B, S, T, D) -> first batch, first sample -> (T, D)
            if "pred_emb" in bd:
                pred_list.append(bd["pred_emb"][0, 0].cpu())
            # Reference embeddings for this env's horizon: (T, D)
            if "z_ref" in bd:
                ref_list.append(bd["z_ref"].cpu())

        total = tracking + terminal + action
        self.tracking_costs.append(tracking)
        self.terminal_costs.append(terminal)
        self.action_costs.append(action)
        self.total_costs.append(total)

        if pred_list:
            self.pred_embeddings.append(torch.stack(pred_list))  # (n_envs, T, D)
        if ref_list:
            self.ref_embeddings.append(ref_list)  # list of n_envs (T, D) tensors

        # Current ref steps per env (these ARE per-env, from solver directly)
        if solver._env_ref_steps is not None:
            self.ref_steps.append(solver._env_ref_steps.copy())


def _decode_embeddings(model, embeddings: torch.Tensor, device: torch.device) -> np.ndarray:
    """Decode latent embeddings to pixel images.

    Args:
        model: World model with decoder attached.
        embeddings: (T, D) latent embeddings.
        device: torch device.

    Returns:
        (T, H, W, 3) uint8 RGB images.
    """
    from stable_pretraining import data as dt

    imagenet_mean = torch.tensor(dt.dataset_stats.ImageNet["mean"], device=device).view(1, -1, 1, 1)
    imagenet_std = torch.tensor(dt.dataset_stats.ImageNet["std"], device=device).view(1, -1, 1, 1)

    with torch.no_grad():
        emb = embeddings.to(device)
        decoded = model.decoder(emb)  # (T, C, H, W)
        decoded = decoded * imagenet_std + imagenet_mean
        decoded = decoded.clamp(0, 1)
        decoded = (decoded * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
    return decoded


def _resize_frame(frame: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize HWC uint8 frame to (h, w, C) using bilinear interpolation."""
    if frame.shape[0] == h and frame.shape[1] == w:
        return frame
    t = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).byte().numpy()


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    """Render matplotlib figure to (H, W, 3) uint8 RGB array."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )
    return arr[:, :, :3].copy()


def render_debug_video(
    debug_logger: DebugLogger,
    video_frames: np.ndarray,       # (n_envs, eval_budget, H, W, C)
    target_frames: list,            # list of (T_i, H, W, C) per env
    references: list,               # list of (N, D) reference embeddings
    env_ref_indices: np.ndarray,    # (n_envs,) mapping env to reference
    rewards: np.ndarray,            # (n_envs, eval_budget)
    episode_successes: np.ndarray,  # (n_envs,)
    video_path: str,
    model=None,                     # world model with decoder for embedding decoding
    action_block: int = 1,          # action_block for computing per-env GT frame indices
    horizon: int = 5,               # planning horizon T
    device: torch.device | None = None,
    fps: int = 10,
) -> None:
    """Render per-env composite debug videos after evaluation."""
    video_path_obj = Path(video_path)
    video_path_obj.mkdir(parents=True, exist_ok=True)

    n_envs = debug_logger.n_envs
    n_steps = len(debug_logger.tracking_costs)
    eval_budget = video_frames.shape[1]

    if n_steps == 0:
        print("[debug_video] No debug data collected, skipping debug video.")
        return

    has_decoder = model is not None and getattr(model, "decoder", None) is not None
    if device is None and model is not None:
        device = next(model.parameters()).device

    # -- Fit PCA on all reference embeddings -----------------------------------
    all_ref_embs = []
    for ref_emb in references:
        if torch.is_tensor(ref_emb):
            all_ref_embs.append(ref_emb.cpu().numpy())
        else:
            all_ref_embs.append(np.array(ref_emb))
    if all_ref_embs:
        all_concat = np.concatenate(all_ref_embs, axis=0)
        pca = PCA(n_components=2)
        pca.fit(all_concat)
    else:
        pca = None

    # -- Pre-decode static GT/recon per env + dynamic pred per step -------------
    # Static: gt_frames_per_env[env_i] = (H, h, w, 3) uint8 — fixed reference frames
    #         recon_per_env[env_i] = (H, h, w, 3) uint8 — decoded ref embeddings
    # Dynamic: decoded_preds[t][env_i] = (H, h, w, 3) uint8 — changes each step
    gt_frames_per_env = {}    # env_i -> (H, h, w, 3)
    recon_per_env = {}        # env_i -> (H, h, w, 3)
    gt_frame_indices = {}     # env_i -> list of int (frame indices for labels)
    decoded_preds = []

    if has_decoder:
        print("[debug_video] Decoding embeddings with decoder ...")

        # Static: GT + recon (once per env, using evenly spaced reference frames)
        for env_i in range(n_envs):
            ref_idx = int(env_ref_indices[env_i]) if env_ref_indices is not None else 0
            ref_emb_tensor = references[ref_idx]  # (N, D)
            if torch.is_tensor(ref_emb_tensor):
                ref_emb_tensor_cpu = ref_emb_tensor.cpu()
            else:
                ref_emb_tensor_cpu = torch.from_numpy(np.array(ref_emb_tensor))

            N = ref_emb_tensor_cpu.shape[0]
            target_len_i = len(target_frames[env_i])

            # Pick H evenly spaced indices through the reference
            indices = np.linspace(0, N - 1, horizon, dtype=int)
            gt_frame_indices[env_i] = indices.tolist()

            # GT: raw pixel frames at those indices
            gt_imgs = []
            for idx in indices:
                # Map embedding index back to frame index (they're 1:1 in set_reference_episodes)
                frame_idx = min(idx, target_len_i - 1)
                gt_imgs.append(target_frames[env_i][frame_idx])
            gt_frames_per_env[env_i] = gt_imgs

            # Recon: decode the reference embeddings at the same indices
            ref_slice = ref_emb_tensor_cpu[indices]  # (H, D)
            recon_per_env[env_i] = _decode_embeddings(model, ref_slice, device)

        # Dynamic: pred per step
        for t in range(n_steps):
            preds_t = {}
            if t < len(debug_logger.pred_embeddings):
                pred_batch = debug_logger.pred_embeddings[t]  # (n_envs, T, D)
                for env_i in range(n_envs):
                    preds_t[env_i] = _decode_embeddings(model, pred_batch[env_i], device)
            decoded_preds.append(preds_t)
        print("[debug_video] Decoding done.")

    # -- Render per-env video --------------------------------------------------
    for env_i in range(n_envs):
        ref_idx = int(env_ref_indices[env_i]) if env_ref_indices is not None else 0
        ref_np = all_ref_embs[ref_idx] if pca is not None else None

        target_len_i = len(target_frames[env_i])

        # Cost arrays for this env: (n_steps,)
        track_arr = np.array([c[env_i] for c in debug_logger.tracking_costs])
        term_arr = np.array([c[env_i] for c in debug_logger.terminal_costs])
        act_arr = np.array([c[env_i] for c in debug_logger.action_costs])
        total_arr = track_arr + term_arr + act_arr

        # PCA-project reference path
        ref_pca = pca.transform(ref_np) if pca is not None and ref_np is not None else None

        # Rewards for this env
        reward_arr = rewards[env_i] if rewards is not None else np.zeros(eval_budget)
        success = bool(episode_successes[env_i]) if episode_successes is not None else False

        out_path = video_path_obj / f"rollout_{env_i}_debug.mp4"
        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264")

        for t in range(eval_budget):
            step_idx = min(t, n_steps - 1)

            # Current ref step for this env
            if debug_logger.ref_steps and step_idx < len(debug_logger.ref_steps):
                cur_ref_step = int(debug_logger.ref_steps[step_idx][env_i])
            else:
                cur_ref_step = t

            # Compute per-env GT frame indices from this env's ref_step + action_block
            ref_frame_idxs = np.array([
                min(cur_ref_step + k * action_block, target_len_i - 1)
                for k in range(horizon)
            ])

            # -- Build figure (960x576 = divisible by 16) ----------------------
            fig = plt.figure(figsize=(12, 7.2), dpi=80)
            fig.patch.set_facecolor("#1a1a2e")

            gs = gridspec.GridSpec(
                3, 3,
                figure=fig,
                hspace=0.40, wspace=0.30,
                left=0.06, right=0.97, top=0.94, bottom=0.08,
                height_ratios=[3, 3, 0.6],
            )

            # -- Row 0: agent frame | expert frame | cost trajectory -----------
            ax_agent = fig.add_subplot(gs[0, 0])
            ax_expert = fig.add_subplot(gs[0, 1])
            ax_cost = fig.add_subplot(gs[0, 2])

            agent_frame = video_frames[env_i, t]
            ax_agent.imshow(agent_frame)
            ax_agent.set_title("Agent Rollout", color="white", fontsize=10, fontweight="bold", pad=4)
            ax_agent.axis("off")

            expert_frame = target_frames[env_i][t % target_len_i]
            ax_expert.imshow(expert_frame)
            ax_expert.set_title("Expert Reference", color="white", fontsize=10, fontweight="bold", pad=4)
            ax_expert.axis("off")

            # Cost trajectory plot
            ts = np.arange(step_idx + 1)
            ax_cost.set_facecolor("#16213e")
            ax_cost.plot(ts, total_arr[:step_idx + 1], color="#ff6b6b", lw=2.0, label="total")
            ax_cost.plot(ts, track_arr[:step_idx + 1], color="#4ecdc4", lw=1.5, label="tracking")
            ax_cost.plot(ts, term_arr[:step_idx + 1], color="#a78bfa", lw=1.5, label="terminal")
            ax_cost.plot(ts, act_arr[:step_idx + 1], color="#fbbf24", lw=1.5, label="action")
            ax_cost.set_xlim(0, max(eval_budget - 1, 1))
            if len(total_arr) > 0:
                y_max = total_arr.max() * 1.15
                ax_cost.set_ylim(0, max(y_max, 0.01))
            ax_cost.set_title("Cost Trajectory", color="white", fontsize=10, fontweight="bold", pad=4)
            ax_cost.set_xlabel("step", color="#cccccc", fontsize=8)
            ax_cost.tick_params(colors="#cccccc", labelsize=8)
            for spine in ax_cost.spines.values():
                spine.set_color("#444477")
            ax_cost.legend(
                fontsize=8, loc="upper right", framealpha=0.7,
                labelcolor="white", facecolor="#1a1a2e", edgecolor="#444477",
            )

            # -- Row 1: 3-row comparison (GT / recon / pred) | latent space ----
            ax_compare = fig.add_subplot(gs[1, :2])
            ax_latent = fig.add_subplot(gs[1, 2])

            ax_compare.set_facecolor("#16213e")
            ax_compare.axis("off")

            # Static GT/recon for this env (doesn't change per step)
            env_gt = gt_frames_per_env.get(env_i)
            env_recon = recon_per_env.get(env_i)
            env_gt_indices = gt_frame_indices.get(env_i, [])
            # Dynamic pred for this step
            pred_imgs = decoded_preds[step_idx].get(env_i) if step_idx < len(decoded_preds) else None

            if has_decoder and env_recon is not None:
                T_h = len(env_gt_indices)
                thumb_h, thumb_w = 112, 112
                gap = 6
                label_h = 16  # space for frame index labels above
                strip_w = T_h * (thumb_w + gap) - gap
                n_rows = 3
                strip_h = n_rows * thumb_h + (n_rows - 1) * gap

                ax_compare.set_xlim(-40, strip_w + 10)
                ax_compare.set_ylim(-4, strip_h + label_h + 4)
                ax_compare.set_title(
                    "GT / Reconstructed (static) / Predicted (dynamic)",
                    color="white", fontsize=10, fontweight="bold", pad=4,
                )

                for j in range(T_h):
                    x = j * (thumb_w + gap)

                    # Frame index label above each column
                    ax_compare.text(
                        x + thumb_w / 2, strip_h + label_h / 2,
                        f"{env_gt_indices[j]}",
                        color="#fbbf24", fontsize=9, fontweight="bold",
                        ha="center", va="center",
                    )

                    # Row 0 (top): GT frames — static, evenly spaced from reference
                    gt_frame = _resize_frame(env_gt[j], thumb_h, thumb_w)
                    y_top = 2 * (thumb_h + gap)
                    ax_compare.imshow(
                        gt_frame,
                        extent=[x, x + thumb_w, y_top, y_top + thumb_h],
                        aspect="auto", zorder=1,
                    )

                    # Row 1 (middle): Reconstructed — static, decoded ref embeddings
                    recon_thumb = _resize_frame(env_recon[j], thumb_h, thumb_w)
                    y_mid = thumb_h + gap
                    ax_compare.imshow(
                        recon_thumb,
                        extent=[x, x + thumb_w, y_mid, y_mid + thumb_h],
                        aspect="auto", zorder=1,
                    )

                    # Row 2 (bottom): Predicted (decoded pred embeddings)
                    if pred_imgs is not None and j < len(pred_imgs):
                        pred_thumb = _resize_frame(pred_imgs[j], thumb_h, thumb_w)
                        ax_compare.imshow(
                            pred_thumb,
                            extent=[x, x + thumb_w, 0, thumb_h],
                            aspect="auto", zorder=1,
                        )

                # Row labels on the left
                label_x = -4
                ax_compare.text(
                    label_x, 2 * (thumb_h + gap) + thumb_h / 2, "GT",
                    color="#ffffff", fontsize=10, fontweight="bold",
                    va="center", ha="right",
                )
                ax_compare.text(
                    label_x, (thumb_h + gap) + thumb_h / 2, "recon",
                    color="#4ecdc4", fontsize=10, fontweight="bold",
                    va="center", ha="right",
                )
                ax_compare.text(
                    label_x, thumb_h / 2, "pred",
                    color="#ff6b6b", fontsize=10, fontweight="bold",
                    va="center", ha="right",
                )
            else:
                ax_compare.set_title(
                    "Decoded Comparison (no decoder loaded)",
                    color="#888888", fontsize=10, pad=4,
                )
                ax_compare.text(
                    0.5, 0.5, "decoder not available",
                    color="#666666", fontsize=12,
                    transform=ax_compare.transAxes, ha="center", va="center",
                )

            # 2D latent space (PCA)
            ax_latent.set_facecolor("#16213e")
            ax_latent.set_title("Latent Space (PCA)", color="white", fontsize=10, fontweight="bold", pad=4)
            ax_latent.tick_params(colors="#cccccc", labelsize=8)
            for spine in ax_latent.spines.values():
                spine.set_color("#444477")

            if pca is not None and ref_pca is not None:
                ax_latent.scatter(
                    ref_pca[:, 0], ref_pca[:, 1],
                    c=np.arange(len(ref_pca)), cmap="cool",
                    s=12, alpha=0.5, zorder=1, label="ref path",
                )
                if cur_ref_step < len(ref_pca):
                    ax_latent.scatter(
                        [ref_pca[cur_ref_step, 0]], [ref_pca[cur_ref_step, 1]],
                        c="cyan", s=60, zorder=3, marker="*",
                        edgecolors="white", linewidths=0.5,
                    )

                if (debug_logger.pred_embeddings
                        and step_idx < len(debug_logger.pred_embeddings)):
                    pred_np = debug_logger.pred_embeddings[step_idx][env_i].numpy()
                    pred_pca = pca.transform(pred_np)
                    ax_latent.plot(
                        pred_pca[:, 0], pred_pca[:, 1],
                        color="#ff6b6b", lw=2.0, zorder=2, alpha=0.9,
                        label="predicted",
                    )
                    ax_latent.scatter(
                        [pred_pca[0, 0]], [pred_pca[0, 1]],
                        c="#ff6b6b", s=40, zorder=4, marker="o",
                        edgecolors="white", linewidths=0.5,
                    )

                ax_latent.legend(
                    fontsize=7, loc="upper left", framealpha=0.7,
                    labelcolor="white", facecolor="#1a1a2e", edgecolor="#444477",
                )

            # -- Row 2: text status bar ----------------------------------------
            ax_status = fig.add_subplot(gs[2, :])
            ax_status.set_facecolor("#0d0d1a")
            ax_status.axis("off")

            cur_total = total_arr[step_idx] if step_idx < len(total_arr) else 0.0
            cur_track = track_arr[step_idx] if step_idx < len(track_arr) else 0.0
            cur_term = term_arr[step_idx] if step_idx < len(term_arr) else 0.0
            cur_act = act_arr[step_idx] if step_idx < len(act_arr) else 0.0
            cur_reward = float(reward_arr[t]) if t < len(reward_arr) else 0.0
            success_str = "YES" if success else "No"

            line1 = (
                f"Step: {t + 1}/{eval_budget}  |  Cost: {cur_total:.3f}  |  "
                f"Reward: {cur_reward:.3f}  |  Success: {success_str}  |  "
                f"Ref step: {cur_ref_step}"
            )
            line2 = (
                f"Tracking: {cur_track:.3f}  |  Terminal: {cur_term:.3f}  |  "
                f"Action: {cur_act:.3f}"
            )
            ax_status.text(
                0.01, 0.78, line1, color="white", fontsize=9, fontweight="bold",
                transform=ax_status.transAxes, va="top",
                fontfamily="monospace",
            )
            ax_status.text(
                0.01, 0.28, line2, color="#cccccc", fontsize=9,
                transform=ax_status.transAxes, va="top",
                fontfamily="monospace",
            )

            # -- Render to RGB array -------------------------------------------
            frame_rgb = _fig_to_rgb(fig)
            plt.close(fig)
            writer.append_data(frame_rgb)

        writer.close()
        print(f"[debug_video] Saved {out_path}")

    print(f"[debug_video] All debug videos saved to {video_path_obj}")
