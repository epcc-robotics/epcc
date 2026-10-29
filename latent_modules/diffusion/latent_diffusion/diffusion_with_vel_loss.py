"""ContinuousDiffusionSDE + MDM/EDGE-style velocity loss on x0-prediction.

Drop-in replacement: same construction args as cleandiffuser
ContinuousDiffusionSDE, plus `lambda_vel` and `lambda_acc`. Forces
`predict_noise=False` so the model's output IS the predicted x0, which is
what the velocity term needs (Δε is meaningless at high noise levels —
cite MDM gauss.py:1334-1339 + EDGE diffusion.py:447-520).

Loss structure (free slots only — fix_mask=1 region is excluded):
    L = L_simple + λ_vel · L_vel + λ_acc · L_acc
where
    L_simple = ‖x0_pred - x0‖²                  (per-frame, unchanged)
    L_vel    = ‖Δx0_pred  - Δx0‖²               (frame-to-frame velocity)
    L_acc    = ‖Δ²x0_pred - Δ²x0‖²              (acceleration, optional)
"""

from __future__ import annotations

import torch

from cleandiffuser.diffusion import ContinuousDiffusionSDE


class VelLossDiffusionSDE(ContinuousDiffusionSDE):
    def __init__(
        self,
        nn_diffusion,
        *,
        lambda_vel: float = 1.0,
        lambda_acc: float = 0.0,
        motion_only_pad: bool = False,
        **kwargs,
    ):
        # Force x0-prediction — vel loss on ε is meaningless (Δε is just noise diff).
        kwargs["predict_noise"] = False
        super().__init__(nn_diffusion, **kwargs)
        self.lambda_vel = float(lambda_vel)
        self.lambda_acc = float(lambda_acc)
        # If True, is_pad masks vel/acc only; L_simple trains on padded freeze
        # targets (DP coverage). Avoids zero-velocity bias from freeze→freeze
        # pairs while keeping late-anchor reconstruction supervision.
        self.motion_only_pad = bool(motion_only_pad)

    def loss(self, x0, condition=None, is_pad=None, **kwargs):
        """Hybrid masked loss.

        Three modes (controlled by `is_pad` and `self.motion_only_pad`):
          1. is_pad=None: no masking (DP, train on all targets).
          2. is_pad given, motion_only_pad=False: ACT mask on all 3 terms.
          3. is_pad given, motion_only_pad=True: L_simple unmasked (DP coverage),
             L_vel/L_acc masked (kills zero-velocity prior from freeze pairs).

        Args:
            x0:        (B, H, D) target
            condition: (B, K*D)
            is_pad:    (B, H) bool — True at clamped/padded positions.
                       None = no masking.
        """
        # Same forward as parent — but pred IS x0 (predict_noise=False).
        xt, t, _ = self.add_noise(x0)
        cond = self.model["condition"](condition) if condition is not None else None
        x0_pred = self.model["diffusion"](xt, t, cond)

        # NOTE: cleandiffuser BaseDiffusionSDE prepends a batch dim to fix_mask
        # and loss_weight (basic.py:70-71). So self.fix_mask is (1, H, D).
        # Slice the H axis with dim=1, NOT dim=0.
        free = 1.0 - self.fix_mask                                     # (1, H, D)

        # is_pad mask broadcast: (B, H, 1). 1 = valid, 0 = padded.
        # valid_motion always reflects is_pad (used by vel/acc).
        # valid_simple reflects is_pad ONLY when motion_only_pad=False;
        # in motion_only_pad=True mode, L_simple sees freeze targets (DP coverage).
        if is_pad is not None:
            valid_motion = (~is_pad.bool()).float().unsqueeze(-1)        # (B, H, 1)
            if self.motion_only_pad:
                valid_simple = torch.ones_like(x0[..., :1])              # (B, H, 1)
            else:
                valid_simple = valid_motion
        else:
            valid_motion = torch.ones_like(x0[..., :1])
            valid_simple = torch.ones_like(x0[..., :1])

        # ---- L_simple (free × valid_simple) ----
        weight_simple = self.loss_weight * free * valid_simple          # (B, H, D)
        denom_simple = weight_simple.sum().clamp(min=1.0)
        loss_simple = ((x0_pred - x0) ** 2 * weight_simple).sum() / denom_simple

        # ---- L_vel (only frame pairs both valid_motion; supervise boundary too) ----
        # Boundary fix: include pair (anchor→pred[1]) where one frame is fix.
        # Original `free * free` excluded boundary, leaving the anchor→first-pred
        # velocity untrained — caused first-step weirdness in sampled trajs.
        # New mask: 0 only when BOTH frames are fix (no info there).
        if self.lambda_vel > 0 and x0.shape[1] >= 2:
            pair_mask = (1.0 - self.fix_mask[:, :-1, :] * self.fix_mask[:, 1:, :]) \
                      * valid_motion[:, :-1, :] * valid_motion[:, 1:, :]  # (B, H-1, D)
            dv_pred = x0_pred[:, 1:] - x0_pred[:, :-1]
            dv_gt = x0[:, 1:] - x0[:, :-1]
            denom = pair_mask.sum().clamp(min=1.0)
            loss_vel = ((dv_pred - dv_gt) ** 2 * pair_mask).sum() / denom
        else:
            loss_vel = x0.new_zeros(())

        # ---- L_acc (triple-frame second difference) ----
        if self.lambda_acc > 0 and x0.shape[1] >= 3:
            triple_mask = (free[:, :-2, :] * free[:, 1:-1, :] * free[:, 2:, :]
                           * valid_motion[:, :-2, :] * valid_motion[:, 1:-1, :] * valid_motion[:, 2:, :])
            da_pred = x0_pred[:, 2:] - 2 * x0_pred[:, 1:-1] + x0_pred[:, :-2]
            da_gt = x0[:, 2:] - 2 * x0[:, 1:-1] + x0[:, :-2]
            denom = triple_mask.sum().clamp(min=1.0)
            loss_acc = ((da_pred - da_gt) ** 2 * triple_mask).sum() / denom
        else:
            loss_acc = x0.new_zeros(())

        return loss_simple + self.lambda_vel * loss_vel + self.lambda_acc * loss_acc
