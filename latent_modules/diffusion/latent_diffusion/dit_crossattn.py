"""DiT1d variant with cross-attention to K-history tokens (Solution C).

Replaces cleandiffuser's DiT1d additive AdaLN conditioning (which compresses
K=6 × D-d history into a single d_model vector via MLPCondition) with
per-token cross-attention. K=6 history is preserved as 6 tokens of d_model
each; each DiT block adds a cross-attention layer where noise tokens (queries)
attend over history tokens (keys/values). Time-step modulation still uses
AdaLN.

Goal: address the conditioning bottleneck identified by 5-agent analysis
(plan: wiggly-riding-falcon.md). MLPCondition compression discards cube-xy
under deployment OOD; cross-attention preserves K-history tokens directly,
allowing each predicted frame to selectively attend to recent history.

Compatibility: DROP-IN replacement for DiT1d in train_cube_velloss.py.
- forward signature: (x, noise, condition) — same as DiT1d.
- condition shape: (B, K * D_history) — flattened, internally reshaped to
  (B, K, D_history).
- Use with `IdentityCondition` from cleandiffuser (no compression).
"""
from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.utils import SinusoidalEmbedding
from cleandiffuser.nn_diffusion import BaseNNDiffusion


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlockCrossAttn(nn.Module):
    """DiT block with self-attention (over noise tokens) + cross-attention
    (noise → K-history tokens) + MLP. AdaLN modulates self-attn and MLP from
    time-step embedding only (history conditioning is now via cross-attn,
    not AdaLN).

    cross_gate=True adds an AdaLN-zero gate on the cross-attn residual so
    cross-attn ramps up from 0 alongside self-attn / MLP gates (otherwise
    cross-attn is at full strength from step 1, while gate_msa/gate_mlp
    start at 0 — asymmetric ramp may destabilize early training dynamics).
    """

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0,
                 cross_gate: bool = False):
        super().__init__()
        self.cross_gate = bool(cross_gate)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, n_heads, dropout, batch_first=True
        )
        self.norm_cross_q = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Cross-attention: queries = noise tokens, keys/values = history tokens.
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, n_heads, dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), approx_gelu(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size))
        # AdaLN modulation. 6 channels by default (shift/scale/gate × 2 for
        # self-attn + MLP). +1 channel (gate_cross) when cross_gate=True so
        # cross-attn residual ramps up from zero like the others.
        n_chunks = 7 if self.cross_gate else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * n_chunks))

    def forward(
        self,
        x: torch.Tensor,
        history: Optional[torch.Tensor],
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:       (B, H, hidden_size) — noise prediction tokens
        history: (B, K, hidden_size) or None — history tokens (cross-attn keys/values)
        t_emb:   (B, hidden_size) — time-step embedding for AdaLN
        """
        if self.cross_gate:
            (shift_msa, scale_msa, gate_msa,
             shift_mlp, scale_mlp, gate_mlp,
             gate_cross) = self.adaLN_modulation(t_emb).chunk(7, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(t_emb).chunk(6, dim=1)
            gate_cross = None
        # Self-attention over noise tokens
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(x_norm, x_norm, x_norm)[0]
        # Cross-attention to history tokens (gated when cross_gate=True).
        if history is not None:
            x_q = self.norm_cross_q(x)
            cross_out = self.cross_attn(x_q, history, history)[0]
            if gate_cross is not None:
                cross_out = gate_cross.unsqueeze(1) * cross_out
            x = x + cross_out
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer1d(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT1dCrossAttn(BaseNNDiffusion):
    """DiT1d with cross-attention to K-history tokens.

    Args identical to DiT1d, plus:
        history_dim: per-token feature dim of the history (e.g. 384 = agent + wrist).
        K: number of history tokens (must match dataset's K).

    Forward signature matches DiT1d:
        x:         (B, H, in_dim) — noise input
        noise:     (B,)            — diffusion timestep
        condition: (B, K * history_dim) — flattened K-history; reshaped to (B, K, D_h)

    Use with `IdentityCondition` so condition flows in unchanged.
    """

    def __init__(
        self,
        in_dim: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
        # New args for cross-attn over K history tokens:
        history_dim: int = 384,
        K: int = 6,
        cross_gate: bool = False,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.in_dim, self.emb_dim = in_dim, emb_dim
        self.d_model = d_model
        self.history_dim = history_dim
        self.K = K

        # Noise-token projection
        self.x_proj = nn.Linear(in_dim, d_model)
        # History-token projection (per-token, no compression)
        self.history_proj = nn.Linear(history_dim, d_model)

        # Time-step embedding
        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.Mish(),
            nn.Linear(d_model, d_model), nn.Mish())

        # Positional embeddings: separate caches for noise (length H) and
        # history (length K). Both sinusoidal; can be replaced with learned later.
        self.pos_emb = SinusoidalEmbedding(d_model)
        self.history_pos_emb = SinusoidalEmbedding(d_model)
        self.pos_emb_cache = None
        self.history_pos_cache = None

        self.blocks = nn.ModuleList([
            DiTBlockCrossAttn(d_model, n_heads, dropout, cross_gate=cross_gate)
            for _ in range(depth)])
        self.final_layer = FinalLayer1d(d_model, in_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Match DiT1d init (Xavier + zeroed AdaLN + zeroed final)
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Time-step MLP
        nn.init.normal_(self.map_emb[0].weight, std=0.02)
        nn.init.normal_(self.map_emb[2].weight, std=0.02)
        # Zero AdaLN
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # Zero final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:         (B, H, in_dim)
        noise:     (B,)
        condition: (B, K * history_dim) — flattened K-history. None ⇒ no cross-attn.
        """
        # Lazy positional cache for noise tokens (length depends on x)
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.pos_emb(torch.arange(x.shape[1], device=x.device))
        # History pos cache (fixed K)
        if self.history_pos_cache is None or self.history_pos_cache.shape[0] != self.K:
            self.history_pos_cache = self.history_pos_emb(
                torch.arange(self.K, device=x.device))

        # Project noise tokens + add pos
        x_proj = self.x_proj(x) + self.pos_emb_cache[None, ]

        # Project history tokens + add pos
        history_tokens = None
        if condition is not None:
            B = condition.shape[0]
            assert condition.shape[1] == self.K * self.history_dim, (
                f"condition shape {condition.shape} mismatch "
                f"K({self.K}) × history_dim({self.history_dim}) = {self.K * self.history_dim}"
            )
            history = condition.reshape(B, self.K, self.history_dim)
            history_tokens = self.history_proj(history) + self.history_pos_cache[None, ]

        # Time-step embedding (AdaLN modulation source for self-attn + MLP)
        t_emb = self.map_noise(noise)
        t_emb = self.map_emb(t_emb)

        # DiT blocks: self-attn + cross-attn + MLP
        for block in self.blocks:
            x_proj = block(x_proj, history_tokens, t_emb)

        # Final layer with time-step AdaLN
        return self.final_layer(x_proj, t_emb)
