"""Any-step Dynamics Model (ADM, Lin et al. 2024 / SuP §4.1) — state-space version.

GRU consumes s_t (encoded to h0) then the action sequence a_t..a_{t+L-1}
sequentially; MLP head on h_L predicts the DELTA state (s_{t+L} - s_t). Predicted
states are NEVER fed back (SuP §4.1) — the hidden state evolves only through actions.
Any-step: L is sampled in [1, Lmax] during training (paper Eq.6 multi-step MSE).

Two tracks share this class, differing only in in/out dims:
  Track A (SuP-faithful, robot-only): state = robot proprio (agent xy+vel)
  Track B (ours, object-aware):       state = full state (agent + block)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ADM(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 512, n_gru: int = 1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.s_enc = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gru = nn.GRU(input_size=action_dim, hidden_size=hidden, num_layers=n_gru, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, state_dim))

    def forward(self, s0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """s0 (B, state_dim) normalized; actions (B, L, action_dim) normalized.
        Returns predicted DELTA state (B, state_dim) normalized (Δ = s_L - s0)."""
        h0 = self.s_enc(s0).unsqueeze(0)                      # (1, B, hidden)
        if self.gru.num_layers > 1:
            h0 = h0.repeat(self.gru.num_layers, 1, 1)
        out, hL = self.gru(actions, h0.contiguous())          # hL (n_gru, B, hidden)
        return self.head(hL[-1])                               # (B, state_dim)

    @torch.no_grad()
    def rollout_multi(self, s0: torch.Tensor, actions: torch.Tensor, ks: list[int]) -> dict[int, torch.Tensor]:
        """For a single start s0 (B,state_dim) and full action seq (B, Lmax, action_dim),
        return {k: predicted Δstate at horizon k} by reading the GRU at each prefix length.
        Directly predicts s_{t+k} from s_t + a_t..a_{t+k-1} (any-step, no bootstrap)."""
        h0 = self.s_enc(s0).unsqueeze(0)
        if self.gru.num_layers > 1:
            h0 = h0.repeat(self.gru.num_layers, 1, 1)
        out, _ = self.gru(actions, h0.contiguous())           # out (B, Lmax, hidden) — h at each step
        res = {}
        for k in ks:
            hk = out[:, k - 1, :]                              # hidden after k actions
            res[k] = self.head(hk)
        return res
