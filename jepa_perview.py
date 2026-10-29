"""Per-View JEPA — multi-token world model without fusion bottleneck.

Key differences from MultiViewJEPA:
  - Each view (agent / wrist / proprio) is encoded into its own 192-d token.
  - Predictor sees N tokens per timestep (interleaved): [A_t, W_t, P_t, A_{t+1}, ...]
  - Block-causal attention mask: same-step tokens see each other; causal across time.
  - View-specific AdaLN action conditioning.
  - Optional: grasp auxiliary head on RAW wrist encoder output (avoids proprio leak).
  - Interface: info['emb'] = concat([A; W; P]) → 192*n_views.

Token streams enabled via config (`tokens.{agent,wrist,proprio}.enabled`).
Supports any non-empty subset for ablation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from jepa import JEPA
from module import FeedForward, modulate


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ProprioEncoder(nn.Module):
    """Small MLP: low-dim proprio (e.g. gripper_width + effector_pos) → 192-d token."""

    def __init__(self, input_dim: int, output_dim: int = 192, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim) or (B*T, input_dim)
        return self.net(x.float())


class GraspAuxHead(nn.Module):
    """Binary classifier reading raw encoder output (cls_emb) for grasp detection.

    Operates on RAW encoder output, NOT predictor output, to avoid proprio leakage
    via attention. Input: (B, T, D). Output: (B, T) logits.
    """

    def __init__(self, input_dim: int = 192, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → (B, T) logits
        return self.net(x).squeeze(-1)


class ViewSpecificAdaLN(nn.Module):
    """View-specific AdaLN-zero modulation. Each view has its own (γ, β, gate) MLP.

    Forward: c (B, N_tokens, D), view_indices (N_tokens,) telling which view each
    token belongs to. Returns 6 * (B, N_tokens, D) chunks.
    """

    def __init__(self, n_views: int, dim: int):
        super().__init__()
        self.n_views = n_views
        self.dim = dim
        self.modules_per_view = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
            for _ in range(n_views)
        ])
        # AdaLN-zero init: start as identity
        for m in self.modules_per_view:
            nn.init.constant_(m[-1].weight, 0)
            nn.init.constant_(m[-1].bias, 0)

    def forward(
        self, c: torch.Tensor, view_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = c.shape
        out = c.new_zeros(B, N, 6 * D)
        for v_idx, module in enumerate(self.modules_per_view):
            sel = (view_indices == v_idx)
            if sel.any():
                out[:, sel] = module(c[:, sel])
        # AdaLN-zero init alone provides the "start as identity" behavior.
        # Do NOT multiply by an external warmup scalar — that would zero out
        # gradient through the gate path and prevent action conditioning from
        # ever developing (silent action-blind failure mode).
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = out.chunk(6, dim=-1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


class PerViewBlock(nn.Module):
    """Transformer block with view-specific AdaLN-zero conditioning + custom mask."""

    def __init__(self, dim: int, n_views: int, heads: int, dim_head: int, mlp_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attn_out = nn.Linear(inner_dim, dim) if not (heads == 1 and dim_head == dim) else nn.Identity()

        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.adaln = ViewSpecificAdaLN(n_views=n_views, dim=dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor, view_indices: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D) where N = T * V (interleaved per timestep)
        c: (B, N, D) per-token conditioning (action embedding broadcast across views)
        view_indices: (N,) which view each token is (0=A, 1=W, 2=P)
        attn_mask: (N, N) bool, True = can attend (block-causal)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(
            c, view_indices,
        )

        # Self-attention with custom mask
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.qkv(x_norm).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        # attn_mask is (N, N) bool — broadcast to (1, 1, N, N) for SDPA
        drop = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),
            dropout_p=drop, is_causal=False,
        )
        attn_out = rearrange(attn_out, "b h n d -> b n (h d)")
        attn_out = self.attn_out(attn_out)
        x = x + gate_msa * attn_out

        # MLP
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_norm)
        return x


class PerViewARPredictor(nn.Module):
    """Autoregressive predictor over per-view tokens.

    Input: (B, T, V, D) — V tokens per timestep
    Output: (B, T, V, D) — V predicted tokens per timestep
    Action conditioning: (B, T, D) — broadcast across V views per step

    Internals:
      - Tokens are interleaved [A_0,W_0,P_0, A_1,W_1,P_1, ...] over N = T*V positions
      - Factored positional embedding: time_pos(T, D) + view_pos(V, D) added per token
      - Block-causal mask: token (t, v) can attend to all (t', v') where t' <= t
      - View-specific AdaLN modulation per block
    """

    def __init__(
        self,
        *,
        history_size: int,
        n_views: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int = 192,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        view_pos_init_scale: float = 1.0,
        attention_mask: str = "block_causal",      # "block_causal" | "strict_causal"
        token_order: str = "interleaved",          # "interleaved" only supported now
        adaln_warmup_steps: int = 0,               # DEPRECATED — kept for config compat
    ):
        super().__init__()
        assert token_order == "interleaved", "Only interleaved token order supported."
        assert attention_mask in ("block_causal", "strict_causal")

        self.T = history_size
        self.V = n_views
        self.D = input_dim
        self.attention_mask_kind = attention_mask
        self.adaln_warmup_steps = 0  # disabled — see ViewSpecificAdaLN.forward

        self.input_proj = nn.Identity() if (input_dim == self.D) else nn.Linear(input_dim, self.D)
        self.output_proj = (nn.Linear(self.D, output_dim)
                            if output_dim is not None and output_dim != self.D else nn.Identity())

        # Factored positional embeddings (additive)
        self.time_pos = nn.Parameter(torch.randn(history_size, self.D))
        self.view_pos = nn.Parameter(torch.randn(n_views, self.D) * view_pos_init_scale)

        self.dropout = nn.Dropout(emb_dropout)

        self.blocks = nn.ModuleList([
            PerViewBlock(dim=self.D, n_views=n_views, heads=heads,
                         dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(self.D)

        # Attention mask and view/time indices are computed DYNAMICALLY per
        # forward call — MPC inference uses T=1 priming (less than history_size),
        # so buffers sized for training-time T would break. See forward() below.

        # Step counter (for AdaLN warmup). Not a parameter.
        self.register_buffer("_step", torch.tensor(0), persistent=False)

    def _gate_scale(self) -> float:
        # Always 1.0 — AdaLN-zero init alone provides smooth start.
        # Multiplying by an external warmup scalar would zero out gradient
        # through the gate path (silent action-blind failure mode).
        return 1.0

    def step_warmup(self):
        """No-op (kept for trainer-side call compatibility)."""
        self._step += 1

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, V, D)  — per-view token stream. T can be <= self.T (training)
                           for MPC inference which primes with T=1.
        c: (B, T, D)     — action embedding per timestep (broadcast across V)
        Returns: (B, T, V, D) — next-step predictions
        """
        B, T, V, D = x.shape
        assert V == self.V and D == self.D, \
            f"Expected V={self.V} D={self.D}, got V={V} D={D}"
        assert T <= self.T, \
            f"T={T} exceeds trained history_size={self.T}"
        device = x.device

        # Flatten to interleaved sequence: (B, T*V, D)
        x_flat = x.reshape(B, T * V, D)
        c_flat = c.unsqueeze(2).expand(B, T, V, D).reshape(B, T * V, D)

        # Dynamic view/time indices (interleaved: token i → time=i//V, view=i%V)
        N = T * V
        view_indices = torch.arange(N, device=device) % V           # (N,)
        time_indices = torch.arange(N, device=device) // V           # (N,)

        # Factored positional embedding: time_pos[t] + view_pos[v] per token
        # Slice time_pos[:T] — only use the first T positions (standard AR pattern)
        pos = self.time_pos[time_indices] + self.view_pos[view_indices]
        x_flat = self.input_proj(x_flat) + pos.unsqueeze(0)
        x_flat = self.dropout(x_flat)

        # Dynamic attention mask
        if self.attention_mask_kind == "block_causal":
            # Token i can attend to j iff time[j] <= time[i] (same-step bidirectional)
            attn_mask = time_indices[:, None] >= time_indices[None, :]
        else:  # strict_causal
            idx = torch.arange(N, device=device)
            attn_mask = idx[:, None] >= idx[None, :]

        for block in self.blocks:
            x_flat = block(x_flat, c_flat, view_indices, attn_mask=attn_mask)
        x_flat = self.norm(x_flat)
        x_flat = self.output_proj(x_flat)

        return x_flat.reshape(B, T, V, D)


# ---------------------------------------------------------------------------
# Main JEPA
# ---------------------------------------------------------------------------

VIEW_NAMES = ("agent", "wrist", "proprio")


class PerViewJEPA(JEPA):
    """JEPA variant emitting per-view tokens (no fusion bottleneck).

    Token order is fixed canonical: agent < wrist < proprio (only enabled ones present).
    """

    def __init__(
        self,
        encoder,                       # weight-tied ViT (used for both A and W)
        predictor: PerViewARPredictor,
        action_encoder,
        projector,                     # ViT_hidden → embed_dim, applied per-view
        pred_proj=None,                # post-predictor projection (per-view)
        decoder=None,
        proprio_encoder: ProprioEncoder | None = None,
        grasp_aux_head: GraspAuxHead | None = None,
        view_config: dict | None = None,
        wrist_key: str = "wrist_pixels",
        proprio_keys: tuple[str, ...] = ("proprio_gripper_opening", "proprio_effector_pos"),
        # grasp label config (used by trainer; stored here for convenience)
        grasp_label_cfg: dict | None = None,
    ):
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj,
            decoder=decoder,
        )
        self.proprio_encoder = proprio_encoder
        self.grasp_aux_head = grasp_aux_head
        self.wrist_key = wrist_key
        self.proprio_keys = tuple(proprio_keys)
        self.grasp_label_cfg = grasp_label_cfg or {}

        # view_config: {'agent': {'enabled': bool, 'loss_weight': float}, ...}
        view_config = view_config or {n: {"enabled": True, "loss_weight": 1.0} for n in VIEW_NAMES}
        self.view_config = view_config

        # Canonical enabled views (preserves order: agent, wrist, proprio)
        self.enabled_views: tuple[str, ...] = tuple(
            v for v in VIEW_NAMES if view_config.get(v, {}).get("enabled", False)
        )
        assert len(self.enabled_views) >= 1, "At least one view must be enabled."
        if grasp_aux_head is not None:
            assert "wrist" in self.enabled_views, "Grasp aux head requires wrist view enabled."

    @property
    def history_size(self) -> int:
        """Number of context frames the predictor was trained with.

        Overrides ``JEPA.history_size`` which reads
        ``self.predictor.pos_embedding`` — ``PerViewARPredictor`` uses
        ``self.T`` / ``self.time_pos`` instead, so the base property would
        raise AttributeError. Returning this correctly is required for
        upstream ``WorldModelPolicy``'s past-action buffer to be sized
        properly (otherwise it falls back to H=1 and never gates on buffer
        fullness, which breaks multi-view warm-up by slicing only 'pixels').
        """
        return int(self.predictor.T)

    # -- helpers --------------------------------------------------------------

    def _encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """ViT encode: (B, T, C, H, W) → (B*T, hidden) cls token."""
        b = pixels.size(0)
        x = rearrange(pixels.float(), "b t ... -> (b t) ...")
        # Pass output_attentions / output_hidden_states explicitly so HF ViT
        # forward never reads `self.config.output_attentions` (broken in
        # transformers 4.46.x — attribute privatised). Semantically identical:
        # we only consume `last_hidden_state[:, 0]` (cls token).
        out = self.encoder(
            x, interpolate_pos_encoding=True,
            output_attentions=False, output_hidden_states=False,
        ).last_hidden_state[:, 0]
        return out, b   # (B*T, hidden), B

    def _encode_proprio(self, info: dict) -> torch.Tensor:
        """Build proprio token by concatenating source columns and running through MLP.
        Returns (B, T, embed_dim).
        """
        parts = []
        for k in self.proprio_keys:
            v = info[k].float()
            if v.dim() == 2:
                v = v.unsqueeze(-1)   # (B, T) → (B, T, 1)
            parts.append(v)
        x = torch.cat(parts, dim=-1)        # (B, T, total_proprio_dim)
        return self.proprio_encoder(x)       # (B, T, embed_dim)

    # -- main forward ---------------------------------------------------------

    def encode(self, info: dict) -> dict:
        """Encode all enabled views into per-view tokens.

        Populates info with:
          - cls_emb_<view>:   (B, T, hidden_or_embed) RAW encoder output per view
          - emb_<view>:       (B, T, embed_dim)       projected per-view token
          - emb:              (B, T, embed_dim * n_views) concat (downstream interface)
          - act_emb:          (B, T, embed_dim) action embedding (if 'action' in info)
        """
        per_view_emb: dict[str, torch.Tensor] = {}
        per_view_cls: dict[str, torch.Tensor] = {}

        if "agent" in self.enabled_views:
            cls_a, b = self._encode_pixels(info["pixels"])
            emb_a = self.projector(cls_a)
            per_view_cls["agent"] = rearrange(cls_a, "(b t) d -> b t d", b=b)
            per_view_emb["agent"] = rearrange(emb_a, "(b t) d -> b t d", b=b)

        if "wrist" in self.enabled_views:
            cls_w, b = self._encode_pixels(info[self.wrist_key])
            emb_w = self.projector(cls_w)
            per_view_cls["wrist"] = rearrange(cls_w, "(b t) d -> b t d", b=b)
            per_view_emb["wrist"] = rearrange(emb_w, "(b t) d -> b t d", b=b)

        if "proprio" in self.enabled_views:
            emb_p = self._encode_proprio(info)
            per_view_emb["proprio"] = emb_p
            per_view_cls["proprio"] = emb_p   # no separate "raw" representation for proprio

        # Store per-view streams under explicit keys
        for v, t in per_view_emb.items():
            info[f"emb_{v}"] = t
        for v, t in per_view_cls.items():
            info[f"cls_emb_{v}"] = t

        # Concat interface (downstream MPC reads this)
        info["emb"] = torch.cat([per_view_emb[v] for v in self.enabled_views], dim=-1)
        # cls_emb keeps the agent cls (or first enabled view) — for backward decoder compat
        first_view = self.enabled_views[0]
        info["cls_emb"] = per_view_cls[first_view]

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, per_view_emb: dict, act_emb: torch.Tensor) -> dict:
        """Run predictor on per-view tokens.

        per_view_emb: dict of view_name → (B, T, D), one entry per enabled view
        act_emb: (B, T, D)
        Returns: dict of view_name → (B, T, D) predicted tokens
        """
        # Stack to (B, T, V, D) in canonical view order
        stacked = torch.stack([per_view_emb[v] for v in self.enabled_views], dim=2)
        preds = self.predictor(stacked, act_emb)   # (B, T, V, D)

        # Apply pred_proj per-view (preserve shape)
        if not isinstance(self.pred_proj, nn.Identity):
            B, T, V, D = preds.shape
            preds_flat = preds.reshape(B * T * V, D)
            preds_flat = self.pred_proj(preds_flat)
            preds = preds_flat.reshape(B, T, V, -1)

        return {v: preds[:, :, i] for i, v in enumerate(self.enabled_views)}

    # -- inference (rollout for MPC) ------------------------------------------

    def rollout(self, info: dict, action_sequence: torch.Tensor, history_size: int = 3) -> dict:
        """Rollout WM autoregressively. action_sequence: (B, S, T_total, action_dim).
        Returns info with `predicted_emb`: (B, S, T_total, embed_dim * n_views) — concat layout.

        Two optimizations (mirroring jepa.py):
        - **Past-action gate**: when `info["_past_action"]` is
          supplied, CEM sampled only FUTURE actions (T-1 chunks). Prepend the
          H-1 real past actions so WM receives true action history.
        - **Pre-computed emb** (commit 8d7eaeec): if `info["emb"]` is already
          populated with 4-D tensor (B, S, T_hist, V*D), skip the vision encoder
          and reuse it — ~10× CEM iteration speedup.
        """
        from jepa import detach_clone

        B, S, T = action_sequence.shape[:3]

        # Pre-encoded path detection (solver.py populates info["emb"] before CEM)
        pre_emb = info.get("emb", None)
        use_precomputed_emb = (
            torch.is_tensor(pre_emb)
            and pre_emb.dim() == 4
            and pre_emb.shape[0] == B
            and pre_emb.shape[1] == S
        )
        if use_precomputed_emb:
            H = pre_emb.shape[2]
        else:
            assert "pixels" in info, "pixels not in info_dict"
            H = info["pixels"].size(2)

        past_action = info.get("_past_action", None)
        if torch.is_tensor(past_action):
            # Strip stray batch/sample dims if past_action arrived as
            # (B, H-1, A) — broadcast to S here.
            if past_action.dim() == 3:
                past_action = past_action.unsqueeze(1).expand(B, S, -1, -1)
            assert past_action.shape[2] == H - 1, (
                f"expected _past_action length {H - 1}, got {past_action.shape[2]}"
            )
            act_first = action_sequence[:, :, :1]
            act_0 = torch.cat([past_action, act_first], dim=2)
            act_future = action_sequence[:, :, 1:]
            n_steps = T - 1
        else:
            # Legacy path: action_sequence carries H past + (T-H) future chunks.
            act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
            n_steps = T - H
        info["action"] = act_0

        if use_precomputed_emb:
            # Unpack concat (..., V*D) → (..., V, D)
            V = len(self.enabled_views)
            D = pre_emb.shape[-1] // V
            assert pre_emb.shape[-1] == V * D, (
                f"pre_emb last dim {pre_emb.shape[-1]} != V({V})*D({D})"
            )
            per_view = pre_emb.reshape(B, S, H, V, D)
            per_view = rearrange(per_view, "b s ... -> (b s) ...").clone()   # (BS, H, V, D)
        else:
            # Encode initial info dict
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init.pop("_past_action", None)
            _init = self.encode(_init)
            per_view_init = torch.stack(
                [_init[f"emb_{v}"] for v in self.enabled_views], dim=2
            )   # (B, H, V, D)
            per_view = per_view_init.unsqueeze(1).expand(B, S, *per_view_init.shape[1:]).clone()
            per_view = rearrange(per_view, "b s ... -> (b s) ...")   # (BS, H, V, D)
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)           # (BS, T_so_far, D)
            x_trunc = per_view[:, -HS:]                  # (BS, HS, V, D)
            c_trunc = act_emb[:, -HS:]                   # (BS, HS, D)
            pred = self.predictor(x_trunc, c_trunc)      # (BS, HS, V, D)
            pred_last = pred[:, -1:]                     # (BS, 1, V, D)
            if not isinstance(self.pred_proj, nn.Identity):
                BS_, _, V_, D_ = pred_last.shape
                pred_last = self.pred_proj(pred_last.reshape(-1, D_)).reshape(BS_, 1, V_, -1)
            per_view = torch.cat([per_view, pred_last], dim=1)
            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        # Final step
        act_emb = self.action_encoder(act)
        x_trunc = per_view[:, -HS:]
        c_trunc = act_emb[:, -HS:]
        pred = self.predictor(x_trunc, c_trunc)
        pred_last = pred[:, -1:]
        if not isinstance(self.pred_proj, nn.Identity):
            BS_, _, V_, D_ = pred_last.shape
            pred_last = self.pred_proj(pred_last.reshape(-1, D_)).reshape(BS_, 1, V_, -1)
        per_view = torch.cat([per_view, pred_last], dim=1)

        # Slice past-context frames off so predicted_emb[:, 0] == o_t, matching
        # the cost model's reference-indexing convention (mirrors jepa.py fix).
        # Without this, predicted_emb[0] is an observation from t-(H-1)*action_block
        # (the past), but cost compares it against ref[step+0] (the current ref) —
        # causing temporal misalignment and garbage cost signal. With H=1 this
        # slice is a no-op (preserves the original behavior for H=1 configs).
        per_view = per_view[:, H - 1:, :, :]

        # Unflatten and concat to (B, S, T_total, V*D) — concat layout for downstream
        per_view = rearrange(per_view, "(b s) t v d -> b s t (v d)", b=B, s=S)
        info["predicted_emb"] = per_view
        return info

    # -- grasp aux ------------------------------------------------------------

    def compute_grasp_label(self, info: dict) -> torch.Tensor | None:
        """Build grasp label on-the-fly from privileged columns.
        Returns (B, T) float in {0, 1} or None if config missing.
        """
        cfg = self.grasp_label_cfg
        if not cfg:
            return None
        contact = info[cfg["contact_key"]].float()
        if contact.dim() == 3:
            contact = contact.squeeze(-1)
        contact_thresh = float(cfg.get("contact_threshold", 0.5))

        eff_key, cube_key = cfg["proximity_pair"]
        eff = info[eff_key].float()        # (B, T, 3)
        cube = info[cube_key].float()      # (B, T, 3)
        dist = torch.linalg.norm(cube - eff, dim=-1)   # (B, T)
        prox_thresh = float(cfg.get("proximity_threshold", 0.05))

        label = ((contact > contact_thresh) & (dist < prox_thresh)).float()
        return label
