"""MPCC cost model: contour error + progress reward + action penalty.

CEM samples augmented action candidates ``(u_phys; ν)`` of shape
``(B, S, H, A_phys + 1)``. The ν slot is a virtual velocity (already
clamped + initialized in CEM via ``physical_dim/nu_clamp/nu_init``;
this cost layer does NOT shape ν further).

Speed semantics (set by ``nu_scale = action_block / (c · ref.N)``):
  - ν=1.0 → natural pace (WM-trained ceiling)
  - ν=c   → expert pace (slow baseline)
  - ν=0.5 → midpoint
  - ν=0   → no progress (idle, allowed)

Cost = q_c · Σ ‖ẑ_k − z_ref(θ_k)‖²  −  q_p · Σ ν_k  +  q_u · Σ ‖u‖²
"""

from __future__ import annotations

import os
import torch
from torch import nn

from latent_modules.mpcc.reference import ReferenceTrajectory


class MPCCCostModel(nn.Module):
    """MPCC cost over an augmented action space [physical_actions, ν].

    Per-env reference handling (mirror of ``latent_modules.mpc.cost``): each env in
    the batch may track a different reference. We group envs by unique
    ``_ref_idx`` and apply each ref's projection / interpolation to its
    masked sub-batch. Avoids the prior env-0-only bug.
    """

    #: [EPCC-5] When True, ``get_cost`` stashes per-horizon-step tensors in
    #: ``self._last_per_step`` so a caller can re-score candidates under a
    #: different horizon mask (e.g. J_trunc = padding excluded). Default False
    #: keeps every existing cell bit-exact.
    emit_per_step: bool = False

    #: [EPCC-5 G1] When True, ``get_cost`` also stashes ``self._last_probe`` with
    #: the projected θ̂ of every rollout step and its ⊥ distance to the contour.
    emit_probe: bool = False

    def __init__(
        self,
        world_model: nn.Module,
        references: list[ReferenceTrajectory],
        physical_action_dim: int,
        action_block: int,
        c: float = 0.3,
        stride: int = 1,
        q_contour: float = 1.0,
        q_progress: float = 0.05,
        q_action: float = 0.01,
        q_action_smooth: float = 0.0,
        action_smooth_all_dims: bool = False,
        z_norm: bool = False,
        horizon_weight: str = "uniform",
        contour_mode: str = "legacy",
        q_lag: float = 0.0,
        ref_target: str = "interp",
        theta_anchor: str = "project",
        q_rot: float = 0.0,
        q_angvel: float = 0.0,
        rot_oracle_gain: float = 0.0,
        rot_probe_path: str = "",
        q_agent: float = 0.0,
        agent_probe_path: str = "",
        q_block: float = 0.0,
        block_probe_path: str = "",
        q_blockvel: float = 0.0,
    ):
        """
        Args:
            action_block: env-step grouping per CEM step (cube AW: 5).
            c: expert speed ratio vs WM training speed (cube AW: 0.3).
                Determines nu_scale so that ν=1.0 maps to natural pace.
            stride: env-step span per reference frame (default 1 for dataset
                refs where 1 frame = 1 env-step; set to diffusion stride S
                for diffusion-MPCC where 1 latent waypoint spans S env-steps).
                Effectively scales the path "logical length" so that ν=1.0
                still means natural pace regardless of waypoint density.
            physical_action_dim: flat phys action dim (= phys_per_step ·
                action_block, e.g., 5 · 5 = 25 for cube AW).
            q_action_smooth: Weight on temporal-smoothness penalty
                ``Σ_t (a_{t+1} − a_t)²`` over the (H · action_block) consecutive
                env-step phys commands within the horizon. Penalises rapid
                changes; the expert is temporally smooth, so non-zero
                ``q_action_smooth`` aligns CEM with that prior. Default ``0.0``
                disables the term entirely (no-op branch).
            action_smooth_all_dims: Toggle that selects which phys dims the
                smoothness penalty applies to.
                  - ``False`` (default): gripper dim only (last entry of each
                    phys-per-step block). Targets the bimodal flicker pattern.
                  - ``True``: ALL phys dims (xyz + yaw + gripper). Targets
                    overall arm trembling + grip-position alignment. Magnitude
                    is ``phys_per_step`` (=5 for cube) × larger than gripper-
                    only at the same weight, so re-tune ``q_action_smooth``
                    accordingly (start ~5× lower).
        """
        super().__init__()
        self.world_model = world_model
        self.references = nn.ModuleList(references)
        self.physical_action_dim = physical_action_dim
        self.action_block = action_block
        self.c = c
        self.stride = int(stride)
        self.q_contour = q_contour
        self.q_progress = q_progress
        self.q_action = q_action
        self.q_action_smooth = float(q_action_smooth)
        self.action_smooth_all_dims = bool(action_smooth_all_dims)
        # ① cost-sharpening knobs ported from MPC (limit WM-depth exposure /
        # rebalance latent dims). z_std computed lazily from the first z_ref.
        self.z_norm = bool(z_norm)
        self.horizon_weight = str(horizon_weight)
        self._z_std = None
        # contour_mode: "legacy" = single full ‖z_pred−z_ref(θ)‖² (=contour²+lag²
        # combined, the original). "split" = proper MPCC decomposition into
        # contour (⊥ to path tangent) + lag (∥ to tangent), weighted separately
        # (q_contour, q_lag). q_lag≫q_contour couples θ to actual progress and
        # stops the progress/agent runaway. Default legacy → bit-exact.
        self.contour_mode = str(contour_mode)
        self.q_lag = float(q_lag)
        # ③ "reachable reference" toggles — the decisive MPC-vs-MPCC test.
        # ref_target: how z_ref(θ) is sampled.
        #   "interp" (default) = linear blend of demo frames (legacy; off-manifold
        #            for non-integer θ → nothing in the data reaches it).
        #   "snap"   = nearest REAL demo waypoint (on-manifold, reachable by an
        #            in-dist action — like MPC's exact-frame ref).
        # theta_anchor: how θ_0 (start of the per-solve ref window) is set.
        #   "project" (default) = project(cur_emb) onto path (legacy; geometry-
        #            driven, can snap anywhere under drift).
        #   "time"   = env-time pointer (`_ref_step` from the solver), like MPC's
        #            time-indexed reference. Needs the solver to advance _ref_step.
        # Compose: project+interp=legacy MPCC; project+snap isolates the
        # off-manifold interpolation; time+interp isolates projection drift;
        # time+snap ≈ MPC. Defaults → bit-exact legacy.
        self.ref_target = str(ref_target)
        self.theta_anchor = str(theta_anchor)
        # ── ROTATION cost (PREVENTION): penalize block-angle mismatch between
        # predicted and reference latents, read via a frozen linear probe
        # (latent → [sin,cos]). Rotation is the dominant stall axis but is buried
        # in the full-L2 contour; this term surfaces it (near-horizon weighted).
        # Default q_rot=0 → term not computed → bit-exact.
        self.q_rot = float(q_rot)
        # q_angvel: penalize predicted block ANGULAR VELOCITY (|Δangle| between
        # consecutive predicted latents, squared via 1−cosΔ). Targets the
        # rotational-MOMENTUM failure (block spins fast under slippery damping →
        # overshoots past recoverable). Discourages fast spin while allowing slow
        # alignment. Reuses the same rot probe. Default 0 → bit-exact.
        self.q_angvel = float(q_angvel)
        self.rot_oracle_gain = float(rot_oracle_gain)
        self._rot_W = None
        self._rot_mlp = None
        if (self.q_rot > 0.0 or self.q_angvel > 0.0 or self.rot_oracle_gain > 0.0) and rot_probe_path:
            import numpy as _np
            _d = _np.load(rot_probe_path)
            if "W1" in _d.files:   # MLP probe: relu(z@W1+b1)@W2+b2 → [sin,cos]
                self._rot_mlp = [torch.tensor(_d[k], dtype=torch.float32)
                                 for k in ("W1", "b1", "W2", "b2")]
                kind = "MLP"
            else:                  # linear probe: z@W+b → [sin,cos]
                self._rot_W = torch.tensor(_d["W"], dtype=torch.float32)
                self._rot_b = torch.tensor(_d["b"], dtype=torch.float32)
                kind = "linear"
            print(f"[MPCC] rotation cost ENABLED q_rot={self.q_rot} q_angvel={self.q_angvel} "
                  f"probe={rot_probe_path} ({kind})")
        # ── AGENT-POSITION cost: penalize decoded agent (pusher) position
        # mismatch between predicted and reference latents, via a frozen linear
        # probe (latent → normalized agent xy). Targets the documented failure
        # where the agent parks away from the block (agent under-encoded in the
        # latent). Agent pos is the controller's OWN state (NOT block GT) → not
        # cheating. Default q_agent=0 → term not computed → bit-exact.
        self.q_agent = float(q_agent)
        self._agent_W = None
        if self.q_agent > 0.0 and agent_probe_path:
            import numpy as _np
            _da = _np.load(agent_probe_path)
            self._agent_W = torch.tensor(_da["W"], dtype=torch.float32)
            self._agent_b = torch.tensor(_da["b"], dtype=torch.float32)
            print(f"[MPCC] agent-position cost ENABLED q_agent={self.q_agent} probe={agent_probe_path}")
        # ── BLOCK-STATE tracking + velocity (2D state-space tracking, default OFF
        # → bit-exact). q_block tracks decoded block(pred) vs block(ref); q_blockvel
        # penalizes predicted block SPEED (anti over-push / slingshot). Block pos is
        # the WM's own decode (probe) + the GIVEN reference → deployable, not cheating.
        self.q_block = float(q_block)
        self.q_blockvel = float(q_blockvel)
        self._block_W = None
        if (self.q_block > 0.0 or self.q_blockvel > 0.0) and block_probe_path:
            import numpy as _np
            _db = _np.load(block_probe_path)
            self._block_W = torch.tensor(_db["W"], dtype=torch.float32)
            self._block_b = torch.tensor(_db["b"], dtype=torch.float32)
            print(f"[MPCC] block-state cost ENABLED q_block={self.q_block} q_blockvel={self.q_blockvel} probe={block_probe_path}")
        # OOD penalties (ported from MPC). Default off → bit-exact baseline.
        # ensemble JRD targets exactly the MPCC failure mode (WM-unreliable
        # actions); density is the kNN p(a|z) variant. Set by the solver.
        self._density_proxy = None
        self._density_lambda = 0.0
        self._ensemble_proxy = None
        self._ensemble_lambda = 0.0

        # Per-call breakdown (mirrors latent_modules.mpc.cost). CEM consumes
        # ``contour_raw`` for unified L2-only logging across MPC/MPCC.
        self._last_breakdown: dict = {}

        # nu_scale per ref: ν=1.0 → natural pace per CEM step.
        # Formula: nu_scale = action_block / (c · stride · ref.N)
        #   - 1 CEM step = action_block env-steps
        #   - Natural pace covers (1/c)× expert distance per env-step
        #   - Path env-step span = stride · ref.N  (stride=1 for dataset
        #     refs; stride=S for diffusion refs at frame stride S)
        #   - ν=1.0 → Δθ = action_block / (c · stride · ref.N) per CEM step
        # For stride=1 (dataset case), reduces to the prior formula.
        self._nu_scales = [action_block / (c * self.stride * ref.N) for ref in references]

    def _rot_sincos(self, z: torch.Tensor) -> torch.Tensor:
        """Latent → [sin,cos] of block angle via the loaded probe (linear or MLP)."""
        if self._rot_mlp is not None:
            W1, b1, W2, b2 = self._rot_mlp
            return (z @ W1 + b1).relu() @ W2 + b2
        return z @ self._rot_W + self._rot_b

    def _agent_xy(self, z: torch.Tensor) -> torch.Tensor:
        """Latent → normalized agent (pusher) xy via the loaded linear probe."""
        return z @ self._agent_W + self._agent_b

    def _block_xy(self, z: torch.Tensor) -> torch.Tensor:
        """Latent → normalized block xy via the loaded linear probe."""
        return z @ self._block_W + self._block_b

    def refresh_reference_trajectories(self, new_references: list[ReferenceTrajectory]) -> None:
        """Hot-swap the per-env reference trajectories and recompute nu_scales.

        Used by diffusion-MPCC's per-replan refresh: every replan, the diffusion
        planner emits a fresh ReferenceTrajectory and we must replace
        ``self.references`` (an nn.ModuleList) AND recompute ``_nu_scales``
        (which depend on ref.N). Also moves the new refs to the cost model's
        device.

        Mirrors ``latent_modules.mpc.MPCCostModel.refresh_references`` pattern but for
        MPCC's arc-length-parameterized refs (ReferenceTrajectory modules).
        Caller (solver) is responsible for resetting ``_env_ref_steps`` and any
        per-env θ-pointer state if the new ref starts from a fresh anchor.

        Args:
            new_references: list[ReferenceTrajectory], one per (env, candidate)
                row. Length must match the original allocation (= n_envs * N).
        """
        if not new_references:
            raise ValueError(
                "refresh_reference_trajectories requires at least one reference"
            )
        # Move all refs to cost model's device (cheap; ref.embeddings already
        # there in the diffusion case, no-op).
        device = next(self.parameters()).device if any(
            True for _ in self.parameters()
        ) else (self.references[0].embeddings.device
                if len(self.references) > 0 else torch.device("cpu"))
        new_refs_on_device = [r.to(device) for r in new_references]
        self.references = nn.ModuleList(new_refs_on_device)
        # Recompute nu_scales — ref.N may differ across new refs (rare but
        # possible for variable-length diffusion outputs).
        self._nu_scales = [
            self.action_block / (self.c * self.stride * r.N)
            for r in new_refs_on_device
        ]

    def _predict_emb(self, rollout_input: dict, physical: torch.Tensor) -> torch.Tensor:
        """Predict the latent rollout for physical action candidates.

        Base impl = the JEPA world-model predictor (free-running rollout).
        Returns predicted_emb shaped (B, S, T, D). Subclasses may override to
        substitute ground-truth env dynamics (oracle diagnostic) while leaving
        the z_ref / ν / contour machinery in ``get_cost`` untouched.
        """
        return self.world_model.rollout(rollout_input, physical)["predicted_emb"]

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Compute MPCC cost.

        ``info_dict`` must contain pre-encoded ``emb`` (4-D, shape
        ``(B, S, T_hist, D)`` populated by ``MPCCSolver.solve``) and
        ``_ref_idx`` shaped ``(B,)`` or ``(B, num_samples)``. Tensors are
        already on device thanks to CEM's batch expansion — only
        ``action_candidates`` and ``_ref_idx`` are explicitly moved.

        ν shaping (clamp / init shift) lives in CEM (`nu_clamp`,
        `nu_init`, `nu_var_scale`). Cost receives ν already in its target
        range and uses it directly.

        Returns ``(B, S)`` cost.
        """
        device = next(self.world_model.parameters()).device
        B = action_candidates.shape[0]
        action_candidates = action_candidates.to(device)

        # CEM broadcasts _ref_idx to (B, num_samples); [:, 0] recovers per-env (B,).
        if "_ref_idx" in info_dict:
            ref_idx = info_dict["_ref_idx"].reshape(B, -1)[:, 0].long().to(device)
        else:
            ref_idx = torch.zeros(B, dtype=torch.long, device=device)

        # Time-anchor: per-env env-step counter (float) for θ_0="time" mode.
        ref_step = None
        if self.theta_anchor == "time" and "_ref_step" in info_dict:
            ref_step = info_dict["_ref_step"].reshape(B, -1)[:, 0].to(device).float()

        # Non-destructive strip — MPPI / multi-iter solvers may reuse info_dict.
        rollout_input = {k: v for k, v in info_dict.items()
                         if k not in ("_ref_idx", "_ref_step")}

        # Split augmented actions → physical + virtual velocity.
        physical = action_candidates[..., :self.physical_action_dim]      # (B, S, H, A_phys)
        nu = action_candidates[..., self.physical_action_dim:].squeeze(-1)  # (B, S, H)

        # Rollout with physical actions only. Factored into _predict_emb so a
        # subclass (e.g. RealEnvMPCCCostModel) can swap the WM predictor for a
        # ground-truth env rollout without touching the z_ref / ν / contour path.
        pred_emb = self._predict_emb(rollout_input, physical)   # (B, S, T, D)
        _B, S, T, D = pred_emb.shape
        assert _B == B, f"predicted_emb batch {_B} != action batch {B}"

        # ── ROTATION-ORACLE (diagnostic upper bound): the WM under-predicts block
        # rotation (~0.58×). De-bias the predicted latent's read-out angle by
        # rot_oracle_gain (≈1/0.58) so the controller "sees" calibrated rotation.
        # Nudges pred_emb along ∂angle/∂z (probe gradient) to set the read-out
        # angle to θ_0 + gain·(θ_pred−θ_0). Tests whether fixing WM rotation
        # prediction would raise SR. Default 0 → off → bit-exact.
        if self.rot_oracle_gain > 0.0 and (self._rot_W is not None or self._rot_mlp is not None):
            if self._rot_mlp is not None and self._rot_mlp[0].device != pred_emb.device:
                self._rot_mlp = [t.to(pred_emb.device) for t in self._rot_mlp]
            if self._rot_W is not None and self._rot_W.device != pred_emb.device:
                self._rot_W = self._rot_W.to(pred_emb.device); self._rot_b = self._rot_b.to(pred_emb.device)
            # analytic ∂angle/∂z (autograd unusable: CEM runs cost in inference_mode)
            if self._rot_mlp is not None:
                W1, b1, W2, b2 = self._rot_mlp                    # (D,H),(H),(H,2),(2)
                pre = pred_emb @ W1 + b1                          # (B,S,T,H)
                mask = (pre > 0).to(pred_emb.dtype)
                h = pre * mask
                sc = h @ W2 + b2                                  # (B,S,T,2)
                s_, c_ = sc[..., 0], sc[..., 1]
                ds = (mask * W2[:, 0]) @ W1.t()                   # ∂s/∂z (B,S,T,D)
                dc = (mask * W2[:, 1]) @ W1.t()                   # ∂c/∂z
            else:
                W, b = self._rot_W, self._rot_b                   # (D,2),(2)
                sc = pred_emb @ W + b
                s_, c_ = sc[..., 0], sc[..., 1]
                ds = W[:, 0].expand_as(pred_emb)
                dc = W[:, 1].expand_as(pred_emb)
            ang = torch.atan2(s_, c_)                             # (B,S,T)
            denom = (s_ * s_ + c_ * c_).clamp(min=1e-8).unsqueeze(-1)
            g = (c_.unsqueeze(-1) * ds - s_.unsqueeze(-1) * dc) / denom   # ∂angle/∂z (B,S,T,D)
            ang0 = ang[:, :, :1]                                  # first-step angle (reference)
            d_ang = ((ang - ang0 + torch.pi) % (2 * torch.pi)) - torch.pi   # wrapped Δ
            target_d = d_ang * float(self.rot_oracle_gain)        # amplified rotation
            delta = (target_d - d_ang)                            # (B,S,T) angle to add
            pred_emb = pred_emb + delta.unsqueeze(-1) * g / (g.pow(2).sum(-1, keepdim=True) + 1e-8)

        # Per-env nu_scale via tensor index. _nu_scales is a Python list of
        # floats; build a (B,) tensor on device once.
        nu_scale_tensor = torch.tensor(
            [self._nu_scales[r] for r in ref_idx.tolist()],
            dtype=nu.dtype, device=device,
        )                                              # (B,)

        # Integrate progress per env:  θ_k = θ_0 + Σ_{j<k} ν_j · nu_scale
        cum_nu = (nu * nu_scale_tensor[:, None, None]).cumsum(dim=-1)   # (B, S, H)

        # Per-env θ_0 via projection + per-env z_ref via interpolate.
        # Group by unique ref_idx to vectorize each unique reference's ops.
        theta_0 = torch.empty(B, S, device=device, dtype=pred_emb.dtype)
        all_theta = torch.empty(B, S, T, device=device, dtype=pred_emb.dtype)
        z_ref = torch.empty(B, S, T, D, device=device, dtype=pred_emb.dtype)
        # local path tangent at each θ (only for contour_mode="split").
        tang = torch.zeros_like(z_ref) if self.contour_mode == "split" else None
        # projected-onto-path latent z(θ̂) (only for contour_mode="project").
        z_proj = torch.zeros_like(z_ref) if self.contour_mode == "project" else None
        # [EPCC-5 G1 probe] θ̂ = where the WM rollout ACTUALLY lands on the contour,
        # and its perpendicular distance. Comparing θ̂ against the ASSIGNED θ
        # answers the geometric half of "is the warm-started chunk consistent with
        # the DTG reference?" — the time half is closed analytically by ν = k·c.
        # Costs one extra projection per candidate, so it is flag-gated and off by
        # default (`contour_mode=legacy` never computes this otherwise).
        _probe = bool(getattr(self, "emit_probe", False))
        probe_theta_hat = torch.empty(B, S, T, device=device, dtype=pred_emb.dtype) if _probe else None
        probe_dist = torch.empty(B, S, T, device=device, dtype=pred_emb.dtype) if _probe else None

        for r in ref_idx.unique().tolist():
            mask = ref_idx == r                       # (B,) bool
            ref_r = self.references[int(r)]

            # θ_0 source (toggle): "time" anchors to the env-step counter (MPC-
            # like, drift-immune); "project" (default) projects the current
            # observation onto the path. All samples share the t=0 latent (cached
            # emb), so [:, 0, 0] picks one representative per env.
            if self.theta_anchor == "time" and ref_step is not None:
                # env_steps_elapsed · Δθ_per_env-step, Δθ_per_env-step =
                # nu_scale / action_block (= 1/(c·N), the natural pace). Matches
                # nu_scale's 1/c so "time" tracks the same pace as ν=1 projection.
                dtheta = float(self._nu_scales[int(r)]) / float(self.action_block)
                t0_theta = (ref_step[mask] * dtheta).clamp(0.0, 1.0)   # (Bg,)
            else:
                t0_emb = pred_emb[mask, 0, 0]          # (Bg, D)
                t0_theta, _ = ref_r.project(t0_emb)    # (Bg,)
            theta_0[mask] = t0_theta[:, None].expand(-1, S).to(theta_0.dtype)

            # Build θ sequence for this ref's envs.
            tg_theta_0 = theta_0[mask]                  # (Bg, S)
            tg_cum_nu = cum_nu[mask]                    # (Bg, S, H)
            tg_all_theta = torch.cat(
                [tg_theta_0[..., None], tg_theta_0[..., None] + tg_cum_nu],
                dim=-1,
            )
            tg_all_theta = tg_all_theta.clamp(0.0, 1.0)
            all_theta[mask] = tg_all_theta.to(all_theta.dtype)

            # Sample ref latents at θ values. ref_target toggle: "snap" returns
            # the nearest REAL demo waypoint (on-manifold, reachable); "interp"
            # (default) linearly blends adjacent frames (legacy, off-manifold).
            _sample = ref_r.snap if self.ref_target == "snap" else ref_r.interpolate
            tg_z_ref = _sample(tg_all_theta)            # (Bg, S, T, D)
            z_ref[mask] = tg_z_ref.to(z_ref.dtype)

            # Local path tangent t(θ) = interp(θ+δ) − interp(θ−δ), δ = one
            # WM-step in θ (= nu_scale), which the stability check found smooth
            # (consecutive-tangent cos ≈ 0.97 at WM-step cadence).
            if tang is not None:
                d = float(self._nu_scales[int(r)])
                tp = ref_r.interpolate((tg_all_theta + d).clamp(0.0, 1.0))
                tm = ref_r.interpolate((tg_all_theta - d).clamp(0.0, 1.0))
                tang[mask] = (tp - tm).to(tang.dtype)

            # Canonical-MPCC projection: θ̂ = closest path point to the PREDICTION;
            # z(θ̂) is the projected latent. contour = ‖z_pred − z(θ̂)‖ (true ⊥
            # distance to path), lag = ‖z(θ̂) − z_ref(θ)‖ (along-path progress gap).
            if _probe:
                _th, _d2 = ref_r.project(pred_emb[mask])       # (Bg, S, T)
                probe_theta_hat[mask] = _th.to(probe_theta_hat.dtype)
                probe_dist[mask] = _d2.clamp(min=0).sqrt().to(probe_dist.dtype)

            if z_proj is not None:
                th_hat, _ = ref_r.project(pred_emb[mask])      # (Bg, S, T)
                z_proj[mask] = ref_r.interpolate(th_hat).to(z_proj.dtype)

        # Cost terms.
        # ① z-score latent normalization (rebalance dims) + ② near-horizon
        # weighting (trust shallow depths where the WM is accurate). Mirrors
        # latent_modules.mpc.cost. Defaults off → bit-exact baseline.
        pe, zr = pred_emb, z_ref
        if self.z_norm:
            if self._z_std is None:
                self._z_std = z_ref.reshape(-1, z_ref.shape[-1]).std(0).clamp(min=1e-6)
            pe = pred_emb / self._z_std
            zr = z_ref / self._z_std
        e = pe - zr                                            # (B, S, T, D)

        # near-horizon weight vector (T,) shared by contour & lag, or None.
        _w = None
        hw = self.horizon_weight
        if hw and hw != "uniform" and e.shape[2] > 1:
            Tloc = e.shape[2]
            if hw == "front":
                _w = torch.arange(Tloc, 0, -1, device=e.device, dtype=e.dtype)
            else:
                vals = [float(x) for x in hw.replace(",", "_").split("_")]
                vals = vals[:Tloc] + [0.0] * max(0, Tloc - len(vals))
                _w = torch.tensor(vals, device=e.device, dtype=e.dtype)
            _w = _w / (_w.sum() + 1e-8) * Tloc

        def _hsum(per_t):                                      # (B,S,T) -> (B,S)
            return (per_t * _w).sum(-1) if _w is not None else per_t.sum(-1)

        if self.contour_mode == "split" and tang is not None:
            # Decompose e onto the path tangent t̂(θ): lag = (e·t̂), contour =
            # perpendicular part. Pythagoras ‖e_⊥‖² = ‖e‖² − (e·t̂)². Weighted
            # separately by q_contour (⊥) and q_lag (∥). z_norm scales the
            # tangent too so the split lives in the cost's metric.
            tv = tang if not self.z_norm else tang / self._z_std
            that = tv / (tv.norm(dim=-1, keepdim=True) + 1e-8)
            e_lag = (e * that).sum(-1)                         # (B,S,T) tangential
            e2 = e.pow(2).sum(-1)                              # (B,S,T) ‖e‖²
            contour_t = (e2 - e_lag.pow(2)).clamp(min=0.0)        # (B,S,T) per-step ⊥
            contour = _hsum(contour_t)                            # ⊥ (B,S)
            lag = _hsum(e_lag.pow(2))                          # ∥ (B,S)
        elif self.contour_mode == "project" and z_proj is not None:
            # exact canonical split via the projection point z(θ̂).
            zp = z_proj if not self.z_norm else z_proj / self._z_std
            contour_t = (pe - zp).pow(2).sum(-1)              # (B,S,T) per-step
            contour = _hsum(contour_t)                        # z_pred → path (B,S)
            lag = _hsum((zp - zr).pow(2).sum(-1))             # path → z(θ) (B,S)
        else:
            contour_t = e.pow(2).sum(-1)                      # (B,S,T) per-step
            contour = _hsum(contour_t)                        # (B,S) legacy full L2
            lag = torch.zeros_like(contour)
        progress = nu.sum(-1)                                   # (B, S)
        action = physical.pow(2).sum((-1, -2))                  # (B, S)

        # REACHABILITY DUMP (gated): per-candidate raw contour + ν so we can ask,
        # per env, whether ANY action in the CEM sampling range reaches the
        # ν-advanced reference (low contour) or all are high (unreachable).
        if os.environ.get("ORACLE_DUMP"):
            import numpy as _np
            _d = os.environ.get("ORACLE_DUMP_DIR", "/tmp/oracle_dump")
            os.makedirs(_d, exist_ok=True)
            _n = getattr(self, "_dumpc", 0)
            if _n < int(os.environ.get("ORACLE_DUMP_N", 40)):
                _np.savez(f"{_d}/dump_{_n:03d}.npz",
                          contour=contour.detach().cpu().numpy(),      # (B,S) raw contour
                          nu=nu.detach().cpu().numpy(),                # (B,S,H) per-step ν
                          progress=progress.detach().cpu().numpy(),   # (B,S) Σν
                          cost=None)
                self._dumpc = _n + 1

        # Temporal smoothness — only computed when enabled. Phys is
        # structured as (B, S, H, ab × pd_per_step). Concatenate consecutive
        # env-step commands across (H · ab) and penalise squared diffs.
        #   - action_smooth_all_dims=False: gripper-only (last entry of pd block)
        #   - action_smooth_all_dims=True:  all phys dims (xyz + yaw + gripper)
        if self.q_action_smooth > 0.0:
            B_, S_, H_, A_phys = physical.shape
            pd_per_step = A_phys // self.action_block
            phys_r = physical.reshape(
                B_, S_, H_, self.action_block, pd_per_step
            )                                                    # (B, S, H, ab, pd)
            if self.action_smooth_all_dims:
                # All-dims: flatten (H, ab) into env-step axis, keep pd dim.
                phys_flat = phys_r.reshape(
                    B_, S_, H_ * self.action_block, pd_per_step
                )                                                # (B, S, H·ab, pd)
                phys_diff = phys_flat[..., 1:, :] - phys_flat[..., :-1, :]
                action_smooth = phys_diff.pow(2).sum((-1, -2))      # (B, S)
            else:
                # Gripper-only: last entry of each pd block, flat across (H, ab).
                grip_seq = phys_r[..., -1]                       # (B, S, H, ab)
                grip_seq_flat = grip_seq.reshape(
                    B_, S_, H_ * self.action_block
                )                                                # (B, S, H·ab)
                grip_diff = grip_seq_flat[..., 1:] - grip_seq_flat[..., :-1]
                action_smooth = grip_diff.pow(2).sum(-1)            # (B, S)
        else:
            action_smooth = torch.zeros_like(progress)

        # ── ROTATION term: probe the block angle from predicted vs reference
        # latents and penalize the circular mismatch (1 − cos Δθ_block),
        # near-horizon weighted (reuse _w). Uses pre-z_norm pred_emb/z_ref so
        # the probe (fit on raw latents) is applied in its native space.
        if self.q_rot > 0.0 and (self._rot_W is not None or self._rot_mlp is not None):
            if self._rot_mlp is not None and self._rot_mlp[0].device != pred_emb.device:
                self._rot_mlp = [t.to(pred_emb.device) for t in self._rot_mlp]
            if self._rot_W is not None and self._rot_W.device != pred_emb.device:
                self._rot_W = self._rot_W.to(pred_emb.device)
                self._rot_b = self._rot_b.to(pred_emb.device)
            sp = self._rot_sincos(pred_emb)                # (B,S,T,2) [sin,cos] of pred
            sr = self._rot_sincos(z_ref)                   # (B,S,T,2) of ref
            sp = sp / (sp.norm(dim=-1, keepdim=True) + 1e-6)
            sr = sr / (sr.norm(dim=-1, keepdim=True) + 1e-6)
            rot_err = 1.0 - (sp * sr).sum(-1)              # (B,S,T) ∈ [0,2], 0=aligned
            rotation = _hsum(rot_err)                       # (B,S) near-horizon weighted
            # angular velocity: 1−cos(Δangle) between consecutive predicted steps
            # (sp already unit). Large = block spinning fast over the horizon.
            if self.q_angvel > 0.0:
                cos_dv = (sp[:, :, 1:, :] * sp[:, :, :-1, :]).sum(-1)   # (B,S,T-1)
                avel = 1.0 - cos_dv                                     # ≥0, fast spin → large
                wv = _w[1:] if _w is not None else None
                angvel = (avel * wv).sum(-1) if wv is not None else avel.sum(-1)  # (B,S)
            else:
                angvel = torch.zeros_like(progress)
        else:
            rotation = torch.zeros_like(progress)
            angvel = torch.zeros_like(progress)

        # ── AGENT-POSITION term: decode agent xy from predicted vs reference
        # latents and penalize the squared mismatch (near-horizon weighted,
        # reuse _w). pre-z_norm space so the probe is applied natively.
        if self.q_agent > 0.0 and self._agent_W is not None:
            if self._agent_W.device != pred_emb.device:
                self._agent_W = self._agent_W.to(pred_emb.device)
                self._agent_b = self._agent_b.to(pred_emb.device)
            ap = self._agent_xy(pred_emb)                   # (B,S,T,2) normalized agent xy
            ar = self._agent_xy(z_ref)                      # (B,S,T,2) of ref
            agent_err = (ap - ar).pow(2).sum(-1)            # (B,S,T)
            agent_track = _hsum(agent_err)                   # (B,S) near-horizon weighted
        else:
            agent_track = torch.zeros_like(progress)

        # ── BLOCK-STATE tracking + velocity (decode block xy from pred/ref).
        if (self.q_block > 0.0 or self.q_blockvel > 0.0) and self._block_W is not None:
            if self._block_W.device != pred_emb.device:
                self._block_W = self._block_W.to(pred_emb.device)
                self._block_b = self._block_b.to(pred_emb.device)
            bp = self._block_xy(pred_emb)                    # (B,S,T,2) normalized block xy
            if self.q_block > 0.0:
                br = self._block_xy(z_ref)
                block_track = _hsum((bp - br).pow(2).sum(-1))
            else:
                block_track = torch.zeros_like(progress)
            if self.q_blockvel > 0.0:                        # penalize predicted block SPEED
                dvb = (bp[:, :, 1:, :] - bp[:, :, :-1, :]).pow(2).sum(-1)   # (B,S,T-1)
                wvb = _w[1:] if _w is not None else None
                block_vel = (dvb * wvb).sum(-1) if wvb is not None else dvb.sum(-1)
            else:
                block_vel = torch.zeros_like(progress)
        else:
            block_track = torch.zeros_like(progress)
            block_vel = torch.zeros_like(progress)

        # [EPCC-5] Per-step tensors for candidate scoring under two rules
        # (J_full = full horizon, J_trunc = padding masked out). Off by default
        # so every existing cell stays bit-exact; SelectMPCCSolver flips it on.
        if _probe:
            self._last_probe = {
                "theta_hat": probe_theta_hat.detach(),   # (B,S,T) where the rollout landed
                "theta_asgn": all_theta.detach(),        # (B,S,T) where we told it to be
                "dist": probe_dist.detach(),             # (B,S,T) ⊥ distance to the contour
            }

        if getattr(self, "emit_per_step", False):
            self._last_per_step = {
                "contour_t": contour_t.detach(),                    # (B,S,T)
                "nu": nu.detach(),                                  # (B,S,H)
                "action_t": physical.pow(2).sum(-1).detach(),       # (B,S,H)
                "physical": physical.detach(),                      # (B,S,H,A_phys)
            }

        # Mirror MPC's breakdown contract for unified CEM logging.
        # ``contour_raw`` is the unweighted L2 term (what MPC calls
        # ``tracking_raw``); CEM reads either to print the L2-only side.
        self._last_breakdown = {
            "contour": (self.q_contour * contour).detach(),        # (B, S) weighted
            "contour_raw": contour.detach(),                        # (B, S) raw L2 sum
            "lag": (self.q_lag * lag).detach(),                     # (B, S) tangential
            "progress": (-self.q_progress * progress).detach(),     # (B, S) signed (reward → −)
            "action": (self.q_action * action).detach(),            # (B, S)
            "action_smooth": (self.q_action_smooth * action_smooth).detach(),
            "rotation": (self.q_rot * rotation).detach(),
            "angvel": (self.q_angvel * angvel).detach(),
            "agent": (self.q_agent * agent_track).detach(),
            "block": (self.q_block * block_track).detach(),
            "blockvel": (self.q_blockvel * block_vel).detach(),
        }

        # env-gated one-shot per-term scale dump (|weighted| median across candidates)
        # to verify no term dominates (e.g. progress not drowned). COST_TERM_DEBUG=1.
        import os as _os
        if _os.environ.get("COST_TERM_DEBUG") and not getattr(self, "_term_dbg_n", 0) >= 3:
            self._term_dbg_n = getattr(self, "_term_dbg_n", 0) + 1
            _m = {k: float(v.abs().float().median()) for k, v in self._last_breakdown.items()
                  if k != "contour_raw"}
            print("[COST_TERMS |weighted| median] " +
                  " ".join(f"{k}={_m[k]:.4g}" for k in _m), flush=True)

        # ─────────────────────────────────────────────────────────────────────
        # COST ASSEMBLY.  The published MPCC objective is exactly THREE terms:
        #     contour (q_contour) + progress reward (q_progress) + action (q_action).
        # Every other term below is an AUXILIARY, EXPERIMENTAL lever added during
        # the lever-finding ablations (see paper appendix / repo history). Each
        # one defaults to weight 0 (or proxy=None), so with the default config the
        # deployed cost is bit-identical to the 3-term objective; an auxiliary term
        # contributes ONLY when a non-default config explicitly sets its weight.
        # They are kept here (rather than deleted) because their plumbing threads
        # through several modules — treat them as OFF for any published result.
        #   EXCEPTION → q_action_smooth: this one is a VALIDATED-EFFECTIVE lever
        #   (a standard temporal-smoothness regularizer, already part of the base
        #   cost formula in the module docstring). It measurably helps on cube and
        #   slippery-pusht (NOT on plain pusht) and is ENABLED in the deployed
        #   recipes (e.g. asm3 + action_smooth_all_dims). It is "0 default" here
        #   only because the weight is set by the recipe config — it is not a
        #   speculative/null lever like the probe/OOD terms below.
        # NOTE: operand order is preserved intentionally (annotations only) so the
        # float reduction stays bit-exact vs the pre-annotation code.
        # ─────────────────────────────────────────────────────────────────────
        cost = (
            self.q_contour * contour                  # [CORE] contouring error ‖z_pred − z_ref(θ)‖
            + self.q_lag * lag                        # [aux] lag (∥); active only when contour_mode=split|project (q_lag=0 default)
            - self.q_progress * progress              # [CORE] progress reward (−Σν)
            + self.q_action * action                  # [CORE] action-magnitude penalty
            + self.q_action_smooth * action_smooth    # [aux ✓ VALIDATED] temporal-smoothness regularizer — effective on cube/slippery (not plain); ON in deployed recipes. (q_action_smooth=0 default, set by recipe)
            + self.q_rot * rotation                   # [aux] rotation-probe cost (q_rot=0 default)
            + self.q_angvel * angvel                  # [aux] angular-velocity cost (q_angvel=0 default)
            + self.q_agent * agent_track              # [aux] agent-position probe cost (q_agent=0 default)
            + self.q_block * block_track              # [aux] block-position probe cost (q_block=0 default)
            + self.q_blockvel * block_vel             # [aux] block-velocity probe cost (q_blockvel=0 default)
        )
        # [aux, EXPERIMENTAL, default-OFF] OOD penalties (proxy=None unless a config
        # enables density/ensemble). Physical action only — strip ν before scoring,
        # the ensemble was trained on z(192)+a_phys(10)=202-d transitions.
        phys_only = action_candidates[..., :self.physical_action_dim]   # (B,S,H,A_phys)
        if self._density_proxy is not None:
            cost = cost + self._density_lambda * self._density_proxy.penalty(phys_only[:, :, 0, :])
        if self._ensemble_proxy is not None:
            cost = cost + self._ensemble_lambda * self._ensemble_proxy.score_batch(pred_emb, phys_only)
        return cost

    def criterion(self, info_dict: dict) -> torch.Tensor:
        """Contour cost from pre-computed embeddings (protocol stub)."""
        pred_emb = info_dict["predicted_emb"]
        _, dist_sq = self.references[0].project(pred_emb)
        return dist_sq.sum(dim=tuple(range(2, dist_sq.ndim)))
