"""DDPM + velocity/acceleration auxiliary loss on x0-space.

Drop-in DDPM subclass that adds:
    L_total = L_simple + λ_vel · L_vel + λ_acc · L_acc

Mirrors `latent_diffusion.diffusion_with_vel_loss.VelLossDiffusionSDE` (the
70% baseline's loss) but on the discrete DDPM ladder instead of continuous
SDE.

Key design decisions:
  - Velocity / acceleration are ALWAYS computed in x0-space, regardless of
    `predict_noise` (the B vs A ablation axis). When `predict_noise=True`,
    we derive `x0_pred = (xt - √(1-ᾱ)·ε_pred) / √ᾱ` and apply the velocity
    formulas on that — same as VelLossDiffusionSDE's principled position.
  - `pair_mask` boundary-fix preserved: anchor→pred[1] pair IS supervised
    (only excludes pairs where BOTH frames are anchor). See line 92 of the
    existing impl + boundary-fix memory.
  - The parent's `add_noise` already applies the anchor-clamp
    `xt = xt*(1-fix_mask) + x0*fix_mask` (ddpm.py:85), so the anchor does
    not leak gradient — no extra clamping needed inside our `loss()`.

Constructor mirrors the cleandiffuser `DDPM` signature plus:
    lambda_vel: float    default 3.0 (matches our 70% baseline's E_v3 setting)
    lambda_acc: float    default 0.0
    motion_only_pad: bool  default False
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

from cleandiffuser.utils import at_least_ndim
from cleandiffuser.diffusion.ddpm import DDPM


class DDPMWithVelLoss(DDPM):
    def __init__(
        self,
        nn_diffusion,
        nn_condition=None,
        *,
        fix_mask=None,
        loss_weight=None,
        lambda_vel: float = 3.0,
        lambda_acc: float = 0.0,
        motion_only_pad: bool = False,
        **kwargs,
    ):
        super().__init__(
            nn_diffusion, nn_condition,
            fix_mask=fix_mask, loss_weight=loss_weight,
            **kwargs,
        )
        self.lambda_vel = float(lambda_vel)
        self.lambda_acc = float(lambda_acc)
        self.motion_only_pad = bool(motion_only_pad)

    # ---------------------------------------------------------------------------
    def _eps_to_x0(self, xt: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor):
        """Recover x0 from a predicted ε using the standard DDPM relation:
            x0 = (xt - √(1-ᾱ) · ε) / √ᾱ
        """
        bar_alpha = at_least_ndim(self.bar_alpha[t], xt.dim())
        return (xt - (1.0 - bar_alpha).sqrt() * eps_pred) / bar_alpha.sqrt().clamp(min=1e-8)

    # ---------------------------------------------------------------------------
    def loss(self, x0, condition=None, is_pad=None, return_x0_pred=False, **kwargs):
        """Hybrid masked loss matching VelLossDiffusionSDE.loss().

        Args:
            x0:        (B, H, D) diffusion target (normalized latents).
            condition: (B, K*D) flat history (passed through nn_condition).
            is_pad:    (B, H) bool — True at padded/clamped positions. None = no mask.

        Three modes (cf. existing impl):
          1. is_pad=None → no mask on any term (DP-coverage default)
          2. is_pad given, motion_only_pad=False → mask all three (ACT-style)
          3. is_pad given, motion_only_pad=True  → mask only vel/acc (DP coverage
             on L_simple, no zero-velocity bias on freeze→freeze pairs)
        """
        xt, t, eps = self.add_noise(x0)
        cond = self.model["condition"](condition) if condition is not None else None
        raw_pred = self.model["diffusion"](xt, t, cond)

        # --- derive x0_pred regardless of prediction mode ---
        if self.predict_noise:
            # B path (ε-pred): recover x0 from the predicted ε.
            x0_pred = self._eps_to_x0(xt, t, raw_pred)
            l_simple_base = (raw_pred - eps) ** 2
        else:
            # A / C path (x0-pred): the network output IS x0.
            x0_pred = raw_pred
            l_simple_base = (raw_pred - x0) ** 2

        # fix_mask shape after DiffusionModel wrap is (1, H, D). Slice on dim=1.
        free = 1.0 - self.fix_mask                                       # (1, H, D)

        # is_pad → (B, H, 1)
        if is_pad is not None:
            valid_motion = (~is_pad.bool()).float().unsqueeze(-1)        # (B, H, 1)
            if self.motion_only_pad:
                valid_simple = torch.ones_like(x0[..., :1])
            else:
                valid_simple = valid_motion
        else:
            valid_motion = torch.ones_like(x0[..., :1])
            valid_simple = torch.ones_like(x0[..., :1])

        # ---- L_simple (free × valid_simple) ----
        weight_simple = self.loss_weight * free * valid_simple
        denom_simple = weight_simple.sum().clamp(min=1.0)
        loss_simple = (l_simple_base * weight_simple).sum() / denom_simple

        # ---- L_vel (frame-to-frame x0 velocity) ----
        # Boundary fix from existing impl: include the anchor→pred[1] pair
        # (mask is 0 only when BOTH frames are fix). Anchor velocity supervised.
        if self.lambda_vel > 0 and x0.shape[1] >= 2:
            pair_mask = (1.0 - self.fix_mask[:, :-1, :] * self.fix_mask[:, 1:, :]) \
                      * valid_motion[:, :-1, :] * valid_motion[:, 1:, :]   # (B, H-1, D)
            dv_pred = x0_pred[:, 1:] - x0_pred[:, :-1]
            dv_gt = x0[:, 1:] - x0[:, :-1]
            denom = pair_mask.sum().clamp(min=1.0)
            loss_vel = ((dv_pred - dv_gt) ** 2 * pair_mask).sum() / denom
        else:
            loss_vel = x0.new_zeros(())

        # ---- L_acc (triple-frame 2nd-difference, optional) ----
        if self.lambda_acc > 0 and x0.shape[1] >= 3:
            triple_mask = (free[:, :-2, :] * free[:, 1:-1, :] * free[:, 2:, :]
                           * valid_motion[:, :-2, :] * valid_motion[:, 1:-1, :]
                           * valid_motion[:, 2:, :])
            da_pred = x0_pred[:, 2:] - 2 * x0_pred[:, 1:-1] + x0_pred[:, :-2]
            da_gt = x0[:, 2:] - 2 * x0[:, 1:-1] + x0[:, :-2]
            denom = triple_mask.sum().clamp(min=1.0)
            loss_acc = ((da_pred - da_gt) ** 2 * triple_mask).sum() / denom
        else:
            loss_acc = x0.new_zeros(())

        total = loss_simple + self.lambda_vel * loss_vel + self.lambda_acc * loss_acc
        if return_x0_pred:
            return total, x0_pred
        return total

    # ---------------------------------------------------------------------------
    def update(self, x0, condition=None, is_pad=None, update_ema=True, **kwargs):
        """Same as DDPM.update but plumbs `is_pad` and returns loss components."""
        loss = self.loss(x0, condition, is_pad=is_pad)
        loss.backward()
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            if self.grad_clip_norm else None
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        if update_ema:
            self.ema_update()
        return {"loss": loss.item(), "grad_norm": grad_norm}
