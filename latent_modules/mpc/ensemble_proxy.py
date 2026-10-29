"""Deploy-time wrapper: scores CEM candidate imagined transitions with the
UNISafe ensemble's JRD epistemic uncertainty and turns it into a hinged penalty
(conformal threshold ε). Faithful to the paper's "detect OOD imaginations":
each imagined (z_t, a_t) along the horizon is scored.

penalty(B,S) = Σ_t max(0, JRD(pred_emb[:,:,t], action_candidates[:,:,t]) − ε)

z = raw JEPA predicted emb (training used raw z); a = scaler-normalized blocked
action (= the CEM candidate action space). Both match training exactly.
"""
import json
from pathlib import Path
import torch

from latent_modules.mpc.ensemble import EnsemblePredictor


class EnsembleProxy:
    def __init__(self, ckpt_path: str, deploy_meta_path: str, device):
        self.ens = EnsemblePredictor.load(ckpt_path, device=device)
        m = json.loads(Path(deploy_meta_path).read_text())
        self.eps = float(m["eps"])
        self.device = device

    @torch.no_grad()
    def score_batch(self, pred_emb: torch.Tensor, action_candidates: torch.Tensor) -> torch.Tensor:
        """pred_emb (B,S,T,D), action_candidates (B,S,H,A) → penalty (B,S)."""
        B, S, T, D = pred_emb.shape
        H = action_candidates.shape[2]
        h = min(H, T)
        z = pred_emb[:, :, :h, :]              # (B,S,h,D)
        a = action_candidates[:, :, :h, :]     # (B,S,h,A)
        u = self.ens.score(z, a)               # (B,S,h)  JRD per imagined transition
        return torch.clamp(u - self.eps, min=0.0).sum(-1)  # (B,S)
