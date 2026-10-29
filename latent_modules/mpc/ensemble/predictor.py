"""Thin wrapper around EnsembleStochasticLinear — mirrors UNISafe's
dreamerv3_torch/uncertainty.py:OneStepPredictor, with the dreamer-specific
`tools.Optimizer` replaced by a plain torch.optim.Adam (same hyperparams:
lr 1e-4, grad_clip 1000). gaussian_nll_loss + the per-member training loop +
intrinsic_reward_penn (JRD) are kept faithful.

Single-step Markov predictor: input = [z_t (D), a_t (A)] → predicts z_{t+1} (D)
as a Gaussian per ensemble member; JRD of the K members = epistemic uncertainty.
"""
import torch
import torch.nn as nn

from .penn import EnsembleStochasticLinear


class EnsemblePredictor(nn.Module):
    def __init__(self, z_dim, a_dim, ensemble_size=5, lr=1e-4, grad_clip=1000.0,
                 device="cuda"):
        super().__init__()
        self.z_dim = z_dim
        self.a_dim = a_dim
        self.ensemble_size = ensemble_size
        in_features = z_dim + a_dim
        # Faithful to UNISafe: hidden_features = in_features, out = z_dim, explore_var='jrd'
        self._networks = EnsembleStochasticLinear(
            in_features=in_features, out_features=z_dim,
            hidden_features=in_features, ensemble_size=ensemble_size,
            explore_var="jrd", residual=True,
        ).to(device)
        self.grad_clip = grad_clip
        self.opt = torch.optim.Adam(self._networks.parameters(), lr=lr)
        self.device = device

    @staticmethod
    def gaussian_nll_loss(mu, target, var):
        # UNISafe uncertainty.py verbatim
        loss = 0.5 * (torch.log(var) + (target - mu) ** 2 / var)
        return torch.mean(loss)

    def train_batch(self, inputs, targets):
        """inputs (N, z+a), targets (N, z). Per-member Gaussian NLL + Adam step
        (mirrors train_ensemble_penn_fixed's per-member loop)."""
        self._networks.train()
        inputs = inputs.to(self.device); targets = targets.to(self.device)
        total = 0.0
        for i in range(self.ensemble_size):
            mu, log_std = self._networks.single_forward(inputs, index=i)  # (N, z)
            var = torch.square(torch.exp(log_std))
            loss = self.gaussian_nll_loss(mu, targets, var)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._networks.parameters(), self.grad_clip)
            self.opt.step()
            total += float(loss.detach())
        return total / self.ensemble_size

    @torch.no_grad()
    def intrinsic_reward_penn(self, inputs):
        """JRD epistemic uncertainty. inputs (N, z+a) or (N,T,z+a) → (N,) or (N,T)."""
        self._networks.eval()
        if inputs.dim() == 3:
            N, T, Dd = inputs.shape
            div = self._networks(inputs.reshape(N * T, Dd))[-1]
            return div.view(N, T)
        return self._networks(inputs)[-1].squeeze(-1)

    @torch.no_grad()
    def score(self, z, a):
        """JRD uncertainty for (z, a). z (...,z_dim), a (...,a_dim) → (...,)."""
        x = torch.cat([z.to(self.device), a.to(self.device)], dim=-1)
        flat = x.reshape(-1, x.shape[-1])
        u = self._networks(flat)[-1].squeeze(-1)  # (M,)
        return u.reshape(x.shape[:-1])

    def save(self, path):
        torch.save({"state_dict": self._networks.state_dict(),
                    "z_dim": self.z_dim, "a_dim": self.a_dim,
                    "ensemble_size": self.ensemble_size}, path)

    @classmethod
    def load(cls, path, device="cuda"):
        ck = torch.load(path, map_location=device, weights_only=False)
        obj = cls(ck["z_dim"], ck["a_dim"], ensemble_size=ck["ensemble_size"], device=device)
        obj._networks.load_state_dict(ck["state_dict"])
        obj._networks.eval()
        return obj
