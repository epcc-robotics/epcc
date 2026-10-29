"""Deploy-side gate for the A^k-conditioned scheduler (paper parameterisation).

`StateADMGate.q_values` builds ONE input `[s0, chunk]` and reads a vector head
indexed by k, matching the shipped `train_iql.py`. The paper's scheduler is
`Q_phi(o, A^k) -> scalar` with `pi(o,A) = argmax_k Q_phi(o, A^k)`, so each
candidate k needs its OWN input vector. This subclass overrides only that —
the ADM, the state extraction, `_merge` and `deviation()` are inherited
unchanged, so the greedy `state_adm` path and the E telemetry are untouched.

The input layout must match build_iql_dataset_ak.py exactly:

    Q input = [ s0 , A^k flattened, zero-padded on the right to chunk*adim ]

`chunk` comes from the training npz via the checkpoint's `sd`
(sd = len(cols) + chunk*adim), so a chunk-length mismatch between training and
deployment is detected rather than silently padded.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from policy.sup_state_gate import StateADMGate, _mlp


class _QNetSA(nn.Module):
    """Twin critics over the state-ACTION vector -> scalar (train_iql_ak.QNetSA)."""
    def __init__(self, sd: int) -> None:
        super().__init__(); self.q1 = _mlp(sd, 1); self.q2 = _mlp(sd, 1)

    def forward(self, x):
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class StateADMGateAK(StateADMGate):
    def __init__(self, *args, iql_path: str | None = None, **kwargs) -> None:
        # Let the parent build the ADM and the state plumbing, but not the Q net —
        # its _QNet has the wrong output width for this architecture.
        super().__init__(*args, iql_path=None, **kwargs)
        self.q = None
        self.policy_q = "min"
        if not iql_path:
            return
        ck = torch.load(iql_path, map_location=self.device, weights_only=False)
        arch = ck.get("arch")
        if arch != "sa_scalar":
            raise ValueError(
                f"{iql_path} has arch={arch!r}, expected 'sa_scalar'. Train it with "
                f"scripts/sup/train_iql_ak.py, or use the stock StateADMGate.")
        self.q_sd = int(ck["sd"])
        self.q = _QNetSA(self.q_sd).to(self.device)
        self.q.load_state_dict(ck["q"]); self.q.eval()
        self.q_sm = np.asarray(ck["s_mean"], np.float32)
        self.q_ss = np.asarray(ck["s_std"], np.float32)
        self.iql_ks = [int(x) for x in ck["KS"]]
        self.policy_q = str(ck.get("policy_q", "min"))
        # sd = len(cols) + chunk*adim  =>  the chunk length the IQL was trained at
        self.q_chunk = (self.q_sd - len(self.cols)) // self.adim
        print(f"[state-gate-ak] Q(o,A^k) scalar  sd={self.q_sd}  KS={self.iql_ks}  "
              f"policy_q={self.policy_q}  train_chunk={self.q_chunk}  eps={ck.get('eps')}")

    @torch.no_grad()
    def q_values(self, info_dict: dict, actions_np: np.ndarray) -> np.ndarray:
        """(N, len(k_values)) — Q(o, A^k) evaluated once per candidate k.

        The caller maps our argmax through the CONFIG's `k_values` list
        (sup_gated.py: `kv[argmax(Q[i])]`), not through the checkpoint's own KS.
        When a scheduler is trained on a restricted candidate set — the paper's
        BiGym row uses k_min=2, k_max=4 — those two lists disagree and the
        executed k would silently be the wrong one. Rather than requiring every
        cell to hand-edit `k_values` to match its checkpoint (a Hydra list
        override that does not survive shell quoting anyway), score only the k
        the checkpoint was trained on and put -inf everywhere else, so the
        caller's argmax can only land on a k this Q net actually knows.
        """
        if self.q is None:
            raise RuntimeError("StateADMGateAK has no Q net (iql_path not given)")
        s0 = self._state(info_dict)
        N, T = actions_np.shape[0], actions_np.shape[1]
        A = actions_np[:, :, :self.adim].astype(np.float32)
        width = self.q_chunk * self.adim
        trained = set(self.iql_ks)
        if not trained.issubset(set(self.k_values)) and not getattr(self, "_warned_ks", False):
            self._warned_ks = True
            print(f"[state-gate-ak] WARN checkpoint KS={self.iql_ks} not all in cell "
                  f"k_values={self.k_values}; those k can never be selected.")
        if T != self.q_chunk and not getattr(self, "_warned_chunk", False):
            self._warned_chunk = True
            print(f"[state-gate-ak] WARN eval chunk T={T} != IQL train chunk "
                  f"{self.q_chunk}; A^k is padded/truncated to {width}. Re-train at "
                  f"the eval chunk length to remove this.")

        nkv = len(self.k_values)
        rows = np.zeros((N, nkv, self.q_sd), np.float32)
        usable = np.zeros(nkv, bool)
        for ci, k in enumerate(self.k_values):
            if k not in trained:
                continue
            merged = self._merge(A[0], k)
            if merged is None:                       # k longer than the chunk
                continue
            mk = np.stack([self._merge(A[e], k) for e in range(N)], axis=0)   # (N, l, adim)
            flat = mk.reshape(N, -1)[:, :width]
            rows[:, ci, :len(self.cols)] = s0
            rows[:, ci, len(self.cols):len(self.cols) + flat.shape[1]] = flat
            usable[ci] = True

        st = torch.as_tensor((rows.reshape(-1, self.q_sd) - self.q_sm) / self.q_ss,
                             dtype=torch.float32, device=self.device)
        q1, q2 = self.q(st)
        if self.policy_q == "mean":
            sc = 0.5 * (q1 + q2)
        elif self.policy_q == "q1":
            sc = q1
        else:
            sc = torch.minimum(q1, q2)
        out = sc.view(N, nkv).cpu().numpy()
        out[:, ~usable] = -np.inf
        return out
