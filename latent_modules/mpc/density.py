"""Deployable density proxy p(a|z) for OOD-action penalisation in CEM-MPC.

Validated in scripts/analysis/validate_density_proxy.py (early k∈[1,3] Spearman
0.60, AUROC 0.87): an action is risky (WM likely to mispredict) when it is far
from the actions the corpus took from nearby states. Built + conformal-calibrated
(threshold q) by scripts/analysis/train_density_proxy.py.

Efficiency: the local action distribution depends only on the CURRENT state z
(fixed per solve), so kNN runs ONCE per solve (set_state); the per-candidate
penalty is a cheap vectorised residual — no kNN inside the CEM loop.
"""
import json
from pathlib import Path
import numpy as np
import torch


class DensityProxy:
    def __init__(self, npz_path: str, meta_path: str, device):
        d = np.load(npz_path)
        m = json.loads(Path(meta_path).read_text())
        self.device = device
        self.ref_zpca = torch.as_tensor(d["ref_zpca"], dtype=torch.float32, device=device)   # (R,pca)
        self.ref_astd = torch.as_tensor(d["ref_astd"], dtype=torch.float32, device=device)   # (R,10)
        self.pca_mean = torch.as_tensor(d["pca_mean"], dtype=torch.float32, device=device)   # (192,)
        self.pca_comp = torch.as_tensor(d["pca_comp"], dtype=torch.float32, device=device)   # (pca,192) whiten-folded
        self.k = int(m["k"]); self.q = float(m["q"]); self.pca_dim = int(m["pca_dim"])
        self._local_mean = None   # (B,10)
        self._local_std = None    # (B,10)

    @torch.no_grad()
    def set_state(self, z_cur: torch.Tensor) -> None:
        """Compute the local corpus-action distribution for each env's current
        state z_cur (B,192). Call ONCE per solve before the CEM iterations."""
        z = z_cur.to(self.device, torch.float32)
        zp = (z - self.pca_mean) @ self.pca_comp.T            # (B,pca) whitened (matches sklearn)
        dist = torch.cdist(zp, self.ref_zpca)                 # (B,R)
        idx = dist.topk(self.k, largest=False).indices        # (B,k) nearest corpus states
        A_nb = self.ref_astd[idx]                             # (B,k,10) their actions
        self._local_mean = A_nb.mean(1)                       # (B,10)
        self._local_std = A_nb.std(1)                         # (B,10)

    @torch.no_grad()
    def penalty(self, a_first: torch.Tensor) -> torch.Tensor:
        """Hinged OOD penalty for candidate first action-blocks a_first (B,S,10),
        in normalized action space (same as ref_astd). Returns (B,S) = max(0, score − q)."""
        if self._local_mean is None:
            return torch.zeros(a_first.shape[:2], device=self.device)
        m = self._local_mean[:, None, :]; s = self._local_std[:, None, :]
        score = torch.linalg.norm((a_first.to(self.device) - m) / (s + 1e-3), dim=-1)  # (B,S)
        return torch.clamp(score - self.q, min=0.0)
