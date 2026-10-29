"""Inverse-dynamics MLP shared between training (`scripts/train_invdyn_cube.py`)
and eval (`eval_diffusion_invdyn.py`).

Maps (z_t, z_{t+stride}) -> action_chunk where z is the LeWM-encoded latent
(192-dim SV cube/PushT / 384-dim MV cube AW) and the action chunk is `stride`
consecutive env-actions stacked: shape (stride, A) for cube (A=5) / PushT (A=2).

history_K=0 (default) is the original (B, A) backward-compat path; history_K>0
prepends K past frames so the model can lean on encoder cur_emb context when
the diffusion ref signal is noisy.
"""

import torch
import torch.nn as nn


class InvDynMLP(nn.Module):
    def __init__(self, z_dim, a_dim, stride=1, hidden=512, add_norm=True,
                 dropout=0.0, history_K=0):
        super().__init__()
        self.z_dim = int(z_dim)
        self.a_dim = int(a_dim)
        self.stride = int(stride)
        self.history_K = int(history_K)
        in_dim = (self.history_K + 2) * self.z_dim
        def block(d_in, d_out):
            return nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.LayerNorm(d_out) if add_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
        self.net = nn.Sequential(
            block(in_dim, hidden),
            block(hidden, hidden),
            nn.Linear(hidden, a_dim * self.stride),
            nn.Tanh(),
        )

    def forward(self, z_hist, z_next):
        """z_hist: (B, K+1, D) or (B, D) for K=0 backcompat.
        z_next : (B, D)."""
        if z_hist.dim() == 2:
            z_hist = z_hist.unsqueeze(1)
        z_flat = z_hist.reshape(z_hist.shape[0], -1)
        out = self.net(torch.cat([z_flat, z_next], dim=-1))
        if self.stride == 1:
            return out
        return out.view(-1, self.stride, self.a_dim)
