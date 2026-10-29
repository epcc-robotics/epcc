"""MPC cost model: time-indexed trajectory tracking in latent space.

Unlike MPCC which uses arc-length parametrization and virtual velocity,
this MPC cost tracks the reference embedding at each planning timestep.

Cost = q_tracking · Σ_k d(ẑ_k, z_ref[k])
     + q_terminal · d(ẑ_T, z_ref[T])
     + q_action · Σ ‖u‖²
     + q_action_smooth · Σ_t ‖a_{t+1} − a_t‖²    (all phys dims, env-step axis)

Distance metrics (configurable):
  l2                – ‖ẑ − z_ref‖²  (sum over embedding dim)
  cosine            – 1 − cos(ẑ, z_ref)
  l2_normalized     – ‖ẑ − z_ref‖² / D  (mean over embedding dim)
  l2_view_weighted  – per-view weighted L2 for per-view concat latents
                      (requires view_dims + view_weights)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def _distance_l2(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance per timestep. (B, S, T)"""
    return (pred - ref).pow(2).sum(-1)


def _distance_cosine(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity per timestep. (B, S, T)"""
    return 1.0 - F.cosine_similarity(pred, ref, dim=-1)


def _distance_l2_normalized(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Mean-squared L2 distance (normalized by embedding dim). (B, S, T)"""
    return (pred - ref).pow(2).mean(-1)


_DISTANCE_FNS = {
    "l2": _distance_l2,
    "cosine": _distance_cosine,
    "l2_normalized": _distance_l2_normalized,
}


def _make_view_weighted_l2(view_dims: list[int], view_weights: list[float]):
    """Factory: per-view weighted L2 for per-view concat latents.

    Splits the embedding's last dim into contiguous chunks sized by
    ``view_dims``, computes squared L2 per view, and weights each by
    ``view_weights``.

    Example (AW with wrist 3×): view_dims=[192, 192], view_weights=[1.0, 3.0]
        cost = ‖A_pred − A_ref‖² + 3.0 · ‖W_pred − W_ref‖²
    """
    assert len(view_dims) == len(view_weights), \
        f"view_dims ({len(view_dims)}) and view_weights ({len(view_weights)}) must match"
    view_dims = list(view_dims)
    view_weights = list(view_weights)

    def fn(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        total = None
        idx = 0
        for dim, w in zip(view_dims, view_weights):
            p = pred[..., idx : idx + dim]
            r = ref[..., idx : idx + dim]
            d = w * (p - r).pow(2).sum(-1)
            total = d if total is None else total + d
            idx += dim
        return total

    return fn


class MPCCostModel(nn.Module):
    """Time-indexed trajectory tracking cost for CEM-based MPC.

    Batched over envs: ``get_cost`` accepts ``_ref_idx`` and ``_ref_step`` of
    shape ``(B,)`` (or ``(B, S)``, first column is used) and gathers per-env
    reference targets in a single vectorized call.
    """

    def __init__(
        self,
        world_model: nn.Module,
        references: list[torch.Tensor],  # list of (N_i, D) embeddings per env
        q_tracking: float = 1.0,
        q_terminal: float = 0.0,
        q_action: float = 0.01,
        q_action_smooth: float = 0.0,
        action_block: int = 1,
        distance: str = "l2",
        view_dims: list[int] | None = None,
        view_weights: list[float] | None = None,
        ref_stride: float | None = None,
        z_norm: bool = False,
        horizon_weight: str = "uniform",
    ):
        """
        Args:
            q_action_smooth: Weight on temporal-smoothness penalty
                ``Σ_t ‖a_{t+1} − a_t‖²`` over the (H · action_block) consecutive
                env-step phys commands within the horizon, summed across ALL
                phys dims (xyz + yaw + gripper). Mirrors latent_mpcc.cost's
                ``q_grip_smooth`` with ``grip_smooth_all_dims=True``: penalises
                arm trembling + grip flicker jointly. Default ``0.0`` disables
                the term entirely. Empirically ``1.0`` works best (per
                MPCC's grip_smooth_all_dims experiments).
        """
        super().__init__()
        self.world_model = world_model
        self.q_tracking = q_tracking
        self.q_terminal = q_terminal
        self.q_action = q_action
        self.q_action_smooth = float(q_action_smooth)
        # ① cost-sharpening knobs (reduce CEM action scatter):
        #   z_norm: per-dim z-score the latent before the distance so small-std,
        #     task-relevant dims aren't swamped by large-std dims (blunt L2 fix).
        #     std computed lazily from the reference embeddings.
        #   horizon_weight: "front" weights near-term steps more (action precision
        #     decides the immediate transition, not the far z_ref baseline).
        self.z_norm = bool(z_norm)
        self.horizon_weight = str(horizon_weight)
        self._z_std = None
        self.action_block = action_block
        # Reference frame stride for tracking. Default = action_block (each
        # planning step looks ``action_block`` env-step ahead in the ref).
        # Override (e.g., ref_stride = action_block / c) lets MPC track a
        # natural-pace ref from a slow c=0.3 dataset by jumping more frames
        # per planning step. ``None`` → action_block (default behavior).
        self.ref_stride = float(ref_stride) if ref_stride is not None else float(action_block)

        # Select distance. ``l2_view_weighted`` is the per-view extension;
        # everything else falls through to the standard table.
        if distance == "l2_view_weighted":
            if not view_dims or not view_weights:
                raise ValueError(
                    "distance='l2_view_weighted' requires view_dims and view_weights"
                )
            self._distance_fn = _make_view_weighted_l2(view_dims, view_weights)
            print(f"[MPCCostModel] per-view weighted L2: "
                  f"dims={view_dims} weights={view_weights}")
        elif distance in _DISTANCE_FNS:
            self._distance_fn = _DISTANCE_FNS[distance]
        else:
            raise ValueError(
                f"Unknown distance '{distance}'. Choose from "
                f"{list(_DISTANCE_FNS)} or 'l2_view_weighted'."
            )

        if not references:
            raise ValueError("MPCCostModel requires at least one reference trajectory")
        for i, r in enumerate(references):
            if r.shape[0] == 0:
                raise ValueError(f"reference {i} is empty (shape {tuple(r.shape)})")

        # Right-pad refs into (R, N_max, D) + per-ref length vector. persistent=False
        # keeps them out of state_dict since references are rebuilt fresh each run.
        padded = pad_sequence([r.detach().cpu() for r in references], batch_first=True)
        lengths = torch.tensor([r.shape[0] for r in references], dtype=torch.long)
        self.register_buffer("_refs_padded", padded, persistent=False)
        self.register_buffer("_ref_lengths", lengths, persistent=False)

        # Captures raw L2 (no q_tracking) for unified CEM init→final logging.
        # Populated in _record_breakdown each get_cost call.
        self._last_breakdown: dict = {}

        # Density-proxy OOD penalty (attached by MPCSolver.configure when enabled).
        # None → no penalty (bit-exact baseline). Set per-solve via the proxy's
        # set_state() in the solver; get_cost adds q_density·max(0, score−q).
        self._density_proxy = None
        self._density_lambda = 0.0
        # Ensemble-JRD OOD penalty (UNISafe port; attached by MPCSolver.configure).
        self._ensemble_proxy = None
        self._ensemble_lambda = 0.0

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Compute trajectory tracking cost.

        ``info_dict`` must contain ``_ref_idx`` and ``_ref_step`` shaped
        ``(B,)`` or ``(B, S)``. Tensors are expected to already be on the
        module's device — the inner solver moves them before the CEM loop.

        Returns ``(B, S)`` cost.
        """
        device = next(self.world_model.parameters()).device
        B = action_candidates.shape[0]

        # The inner solver broadcasts _ref_idx to (B, num_samples); [:, 0] recovers (B,).
        ref_idx = info_dict["_ref_idx"].reshape(B, -1)[:, 0].long()
        # ref_step may be float64 (advance_steps accumulates ref_stride/action_block)
        # — floor (truncate via .long() cast for non-negative values) so the
        # ref window never reads frames the agent hasn't logically reached.
        ref_step = info_dict["_ref_step"].reshape(B, -1)[:, 0].long()

        # MPPI reuses info_dict across iters — non-destructive strip.
        rollout_input = {k: v for k, v in info_dict.items() if k not in ("_ref_idx", "_ref_step")}
        rollout_input = self.world_model.rollout(rollout_input, action_candidates)
        pred_emb = rollout_input["predicted_emb"]  # (B, S, T, D)
        _B, S, T, D = pred_emb.shape
        assert _B == B, f"predicted_emb batch {_B} != action batch {B}"

        refs_b = self._refs_padded[ref_idx]        # (B, N_max, D)
        lengths_b = self._ref_lengths[ref_idx]     # (B,)

        t_offsets = (torch.arange(T, device=device, dtype=torch.float32)
                     * self.ref_stride).long()  # floor (truncate)
        ref_indices = ref_step[:, None] + t_offsets[None, :]
        ref_indices = torch.minimum(ref_indices, (lengths_b - 1)[:, None]).clamp(min=0)

        gather_idx = ref_indices[:, :, None].expand(B, T, D)
        z_ref = torch.gather(refs_b, 1, gather_idx)
        z_ref = z_ref[:, None, :, :].expand(B, S, T, D)

        # ① z-score normalization: rebalance dims so the cost resolves small
        # task-relevant action differences (lazy per-dim std from references).
        if self.z_norm:
            if self._z_std is None:
                rp = self._refs_padded                      # (R, N_max, D)
                rl = self._ref_lengths                      # (R,)
                Nmax = rp.shape[1]
                mask = (torch.arange(Nmax, device=rp.device)[None, :] < rl[:, None])
                valid = rp[mask]                            # (sum_len, D)
                self._z_std = valid.std(0).clamp(min=1e-6)  # (D,)
            pred_emb = pred_emb / self._z_std
            z_ref = z_ref / self._z_std

        dist = self._distance_fn(pred_emb, z_ref)
        # ① horizon weighting: emphasize the depths where the WM is reliable.
        #   "uniform" → plain sum.  "front" → linear [T..1].
        #   "<f0,f1,...>" comma list → arbitrary per-depth weights (e.g. measured
        #   inverse-variance 1/err(h)²; set h=0 → 0 since it carries no action info).
        hw = self.horizon_weight
        if hw and hw != "uniform" and T > 1:
            if hw == "front":
                w = torch.arange(T, 0, -1, device=device, dtype=torch.float32)
            else:
                vals = [float(x) for x in hw.replace(",", "_").split("_")]
                vals = (vals[:T] + [0.0] * max(0, T - len(vals)))
                w = torch.tensor(vals, device=device, dtype=torch.float32)
            w = w / (w.sum() + 1e-8) * T                     # keep total scale ~T
            tracking = (dist * w[None, None, :]).sum(-1)     # (B, S)
        else:
            tracking = dist.sum(-1)                          # (B, S)
        terminal = dist[:, :, -1]                            # (B, S)
        action_cost = action_candidates.pow(2).sum((-1, -2))  # (B, S)

        # Temporal smoothness — all phys dims, env-step axis. ``action_candidates``
        # is structured ``(B, S, H, action_block · pd_per_step)``. Reshape to
        # ``(B, S, H · action_block, pd_per_step)``, squared-diff across the
        # env-step axis, and sum over (pd, env-step). Mirrors MPCC's
        # ``grip_smooth_all_dims=True`` branch.
        if self.q_action_smooth > 0.0:
            B_, S_, H_, A_ = action_candidates.shape
            pd_per_step = A_ // self.action_block
            phys_flat = action_candidates.reshape(
                B_, S_, H_ * self.action_block, pd_per_step
            )                                                # (B, S, H·ab, pd)
            phys_diff = phys_flat[..., 1:, :] - phys_flat[..., :-1, :]
            action_smooth = phys_diff.pow(2).sum((-1, -2))    # (B, S)
        else:
            action_smooth = torch.zeros_like(tracking)

        self._record_breakdown(
            ref_idx=ref_idx,
            tracking=tracking,
            terminal=terminal,
            action_cost=action_cost,
            action_smooth=action_smooth,
            pred_emb=pred_emb,
            z_ref=z_ref,
            ref_indices=ref_indices,
        )

        # ─────────────────────────────────────────────────────────────────────
        # COST ASSEMBLY.  The deployed MPC objective is latent tracking + action
        # magnitude (q_tracking, q_action). q_terminal and q_action_smooth are
        # standard optional regularizers that default to 0.
        #   q_action_smooth is a VALIDATED-EFFECTIVE lever: the temporal-smoothness
        #   penalty measurably helps MPC on cube and slippery-pusht (NOT plain
        #   pusht) and is ENABLED in the deployed recipes — "0 default" only because
        #   the weight is set by the recipe config, NOT a speculative/null lever.
        # By contrast the density / ensemble OOD penalties below are AUXILIARY,
        # EXPERIMENTAL levers added during the lever-finding ablations; both default
        # to proxy=None → OFF, so with the default config the cost is bit-identical
        # to the base objective. They are kept (not deleted) because the proxy
        # plumbing threads through several modules — treat them OFF for publication.
        # ─────────────────────────────────────────────────────────────────────
        cost = (
            self.q_tracking * tracking                # [CORE] latent tracking ‖z_pred − z_ref‖
            + self.q_terminal * terminal              # [reg] terminal tracking (q_terminal=0 default)
            + self.q_action * action_cost             # [CORE] action-magnitude penalty
            + self.q_action_smooth * action_smooth    # [reg ✓ VALIDATED] temporal-smoothness regularizer — effective on cube/slippery (not plain); ON in deployed recipes. (q_action_smooth=0 default, set by recipe)
        )
        # [aux, EXPERIMENTAL, default-OFF] Density-proxy OOD penalty on the first
        # executed action-block (preventive: catch the exploiting action before
        # drift). Hinged at conformal q so in-dist candidates are untouched →
        # tracking unchanged when score ≤ q. proxy=None by default → no-op.
        if self._density_proxy is not None:
            dpen = self._density_proxy.penalty(action_candidates[:, :, 0, :])  # (B,S)
            cost = cost + self._density_lambda * dpen
        # [aux, EXPERIMENTAL, default-OFF] Ensemble-JRD OOD penalty over imagined
        # transitions along the horizon. proxy=None by default → no-op.
        if self._ensemble_proxy is not None:
            epen = self._ensemble_proxy.score_batch(pred_emb, action_candidates)  # (B,S)
            cost = cost + self._ensemble_lambda * epen
        return cost

    def _record_breakdown(self, *, tracking: torch.Tensor, **_kwargs) -> None:
        """Capture raw L2 (no q_tracking) for unified CEM init→final logging.

        ``MPCCostModelDebug`` overrides this for full per-env breakdown
        collection (used by debug video). The base implementation only
        keeps the raw tracking term so CEM's per-env log line can fetch it
        via ``cost_model._last_breakdown['tracking_raw']``.
        """
        self._last_breakdown = {"tracking_raw": tracking.detach()}


class MPCCostModelDebug(MPCCostModel):
    """Cost model that captures per-env breakdowns for debug visualization.

    Adds one Python loop + B GPU→CPU copies per ``get_cost`` call, so only
    use via :class:`~latent_mpc.MPCSolverDebug` when debug_video is enabled.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-env breakdowns keyed by ref_idx (env→ref may be many-to-one when
        # set_reference_episodes dedupes (ep, start, goal) triples).
        self._env_breakdowns: dict[int, dict] = {}

    def _record_breakdown(
        self,
        *,
        ref_idx: torch.Tensor,
        tracking: torch.Tensor,
        terminal: torch.Tensor,
        action_cost: torch.Tensor,
        action_smooth: torch.Tensor,
        pred_emb: torch.Tensor,
        z_ref: torch.Tensor,
        ref_indices: torch.Tensor,
    ) -> None:
        # Populate base ``_last_breakdown[tracking_raw]`` so CEM's per-env
        # log line works in Debug mode too.
        super()._record_breakdown(tracking=tracking)
        ref_idx_list = ref_idx.tolist()                    # single GPU→CPU sync
        ref_indices_cpu = ref_indices.detach().cpu()        # single D2H copy
        for b, r in enumerate(ref_idx_list):
            # Store (1, S) slices on CPU so dict entries don't pin the big
            # (B, S, T, D) parent tensor in VRAM across env steps.
            self._env_breakdowns[r] = {
                "tracking":     (self.q_tracking      * tracking[b : b + 1]).detach().cpu(),
                "terminal":     (self.q_terminal      * terminal[b : b + 1]).detach().cpu(),
                "action":       (self.q_action        * action_cost[b : b + 1]).detach().cpu(),
                "action_smooth":(self.q_action_smooth * action_smooth[b : b + 1]).detach().cpu(),
                "pred_emb":     pred_emb[b : b + 1].detach().cpu(),
                "z_ref":        z_ref[b, 0].detach().cpu(),
                "ref_indices":  ref_indices_cpu[b],
            }
