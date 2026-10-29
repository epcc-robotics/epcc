"""IQL scheduler over Q(o, A^k) — the paper's parameterisation.

Companion to build_iql_dataset_ak.py. The shipped `train_iql.py` learns
Q(s) -> vector-over-k from an input that is identical for every k, so the only
thing distinguishing the six candidates is the output head index. The paper
(Sec 4.2, Eq.9, Algorithm 1 line 26) learns a scalar

    Q_phi(o, A^k)          pi(o, A) = argmax_k Q_phi(o, A^k)
    V_psi(o)               L_V = expectile( V(o) - Q(o, A^k) )
    L_Q = ( r' + gamma * V(o') - Q(o, A^k) )^2

so the merged action IS the input. That is what lets the network learn a rule
that transfers across states instead of memorising a per-state answer.

Mapping onto the npz written by build_iql_dataset_ak.py:
    Q input  <- SQ  = [s0, A^k padded]     the state-action (o, A^k)
    V input  <- S   = [s0, A   padded]     the CMDP state   (o, A)
    V(o')    <- SN  = [sp[k], A_next]      the next CMDP state

Everything else (OMEGA, gamma, expectile tau, step count, twin-Q targets) is
kept at the shipped values so this is a single-variable change against
train_iql.py. The one extra knob is --policy_q, because the shipped file
extracts the policy with the pessimistic min over the twin critics while the
paper's rule is a plain argmax over one Q — with OMEGA=600 that pessimism is
not a detail, so it is made measurable rather than silently changed.

`--eps_abs` mirrors the absolute-threshold flag added to train_iql.py; the
paper's eps is an absolute deviation threshold swept in Sec 5.4, not a quantile.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAU, GAMMA, POLYAK, LR, STEPS, BS, OMEGA = 0.7, 0.99, 0.005, 3e-4, 40000, 512, 600.0


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class QNetSA(nn.Module):
    """Twin critics over the state-ACTION vector (o, A^k) -> scalar."""
    def __init__(s, sd):
        super().__init__(); s.q1 = mlp(sd, 1); s.q2 = mlp(sd, 1)

    def forward(s, x):
        return s.q1(x).squeeze(-1), s.q2(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps_q", type=float, default=0.4)
    ap.add_argument("--eps_abs", type=float, default=None,
                    help="absolute deviation threshold (paper Tab.5 style); overrides --eps_q")
    ap.add_argument("--omega", type=float, default=OMEGA)
    ap.add_argument("--policy_q", choices=["min", "mean", "q1"], default="min",
                    help="how argmax_k reads the twin critics. 'min' is what the shipped "
                         "train_iql.py does; the paper's rule is a single Q, i.e. 'q1'/'mean'.")
    # Appendix G / Table 9 of the paper gives per-benchmark values that our port
    # never used. The gaps are large and all in the conservative direction:
    #                    BiGym   Libero   Real     ours (before)
    #   Penalty (Omega)    -5      -2      -1        -600
    #   Gamma              0.9     0.1     0.9        0.99
    #   Expectile (tau)   0.95    0.95    0.95        0.7
    #   k_min               2       1       2          1
    #   k_max               4       2       4          6
    # k_min=2 is the structural one: with k=1 removed from the candidate set the
    # scheduler CANNOT collapse to "never downsample", which is the failure we
    # have been measuring. Note also that Omega=-5 with gamma=0.9, Kmax=4 does
    # NOT satisfy Prop 4.1's bound (gamma*Kmax/(1-gamma) = 36) — the authors run
    # a much weaker penalty than their own theorem requires, so lowering Omega
    # is returning to the paper, not departing from it.
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--k_min", type=int, default=1)
    ap.add_argument("--k_max", type=int, default=99)
    a = ap.parse_args()

    z = np.load(a.dataset)
    for need in ("S", "SQ", "K", "SN", "DONE", "E", "KS"):
        if need not in z:
            raise SystemExit(f"{a.dataset} has no '{need}' — build it with build_iql_dataset_ak.py")
    S, SQ, K, SN, DONE, E = z["S"], z["SQ"], z["K"], z["SN"], z["DONE"], z["E"]
    KS_ALL = [int(x) for x in z["KS"]]
    # Restrict the candidate set. Rows are emitted state-major / k-minor, so a
    # mask over K keeps the per-state blocks contiguous and the pick% reshape
    # below stays valid.
    KS = [k for k in KS_ALL if a.k_min <= k <= a.k_max]
    if not KS:
        raise SystemExit(f"no k in [{a.k_min}, {a.k_max}] within {KS_ALL}")
    if KS != KS_ALL:
        keep = np.isin(K, KS)
        S, SQ, K, SN, DONE, E = S[keep], SQ[keep], K[keep], SN[keep], DONE[keep], E[keep]
        print(f"[IQLak] candidate set restricted {KS_ALL} -> {KS}  ({keep.sum()} of {len(keep)} rows)")
    nk = len(KS)

    if a.eps_abs is not None:
        eps = float(a.eps_abs)
        q = float((E[K > 1] <= eps).mean())
        print(f"[IQLak] eps ABSOLUTE {eps:.4f} -> D' quantile {q:.3f} "
              f"(--eps_q {a.eps_q} would give {float(np.quantile(E[K > 1], a.eps_q)):.4f})")
    else:
        eps = float(np.quantile(E[K > 1], a.eps_q))
    R = np.where(E <= eps, K.astype(np.float32), -a.omega).astype(np.float32)

    # One scaler for both heads: S, SQ and SN share the same layout and width.
    ALL = np.concatenate([S, SQ, SN], 0)
    sm, ss = ALL.mean(0), ALL.std(0) + 1e-6
    Sn = torch.tensor((S - sm) / ss, device=DEV)
    SQn = torch.tensor((SQ - sm) / ss, device=DEV)
    SNn = torch.tensor((SN - sm) / ss, device=DEV)
    Rt = torch.tensor(R, device=DEV)
    Dt = torch.tensor(DONE, device=DEV)
    N, sd = len(S), S.shape[1]
    print(f"[IQLak] {a.dataset.split('/')[-1]} N={N} sd={sd} eps={eps:.4f} "
          f"safe={np.mean(R > 0) * 100:.0f}% policy_q={a.policy_q} omega={a.omega} gamma={a.gamma} tau={a.tau} KS={KS}")
    for k in KS:
        mk = K == k
        print(f"    k={int(k)}: safe={float((E[mk] <= eps).mean()) * 100:3.0f}%  meanE={float(E[mk].mean()):.4f}")

    q = QNetSA(sd).to(DEV)
    qt = QNetSA(sd).to(DEV); qt.load_state_dict(q.state_dict())
    v = mlp(sd, 1).to(DEV)
    oq = torch.optim.Adam(q.parameters(), LR)
    ov = torch.optim.Adam(v.parameters(), LR)

    for step in range(STEPS):
        ix = torch.randint(0, N, (BS,), device=DEV)
        s, sq, r, sn, dn = Sn[ix], SQn[ix], Rt[ix], SNn[ix], Dt[ix]
        with torch.no_grad():
            q1t, q2t = qt(sq)
            qsa = torch.minimum(q1t, q2t)          # target uses min (standard IQL)
        vs = v(s).squeeze(-1)
        vl = (torch.where(qsa - vs >= 0, a.tau, 1 - a.tau) * (qsa - vs) ** 2).mean()
        ov.zero_grad(); vl.backward(); ov.step()
        with torch.no_grad():
            tgt = r + a.gamma * (1 - dn) * v(sn).squeeze(-1)
        q1, q2 = q(sq)
        ql = ((q1 - tgt) ** 2 + (q2 - tgt) ** 2).mean()
        oq.zero_grad(); ql.backward(); oq.step()
        with torch.no_grad():
            for p, pt in zip(q.parameters(), qt.parameters()):
                pt.data.mul_(1 - POLYAK).add_(POLYAK * p.data)

    # pick%: rows are emitted state-major, k-minor, so reshape recovers the
    # six candidates of each state and argmax over that axis is the policy.
    n_states = min(20000 // nk, N // nk)
    with torch.no_grad():
        sub = SQn[: n_states * nk]
        a1, a2 = q(sub)
        if a.policy_q == "min":
            sc = torch.minimum(a1, a2)
        elif a.policy_q == "mean":
            sc = 0.5 * (a1 + a2)
        else:
            sc = a1
        pick = sc.view(n_states, nk).argmax(1).cpu().numpy()
    dist = {int(KS[i]): round(int((pick == i).sum()) / len(pick) * 100) for i in range(nk)}
    mean_k = sum(int(KS[i]) * int((pick == i).sum()) for i in range(nk)) / len(pick)

    torch.save({"q": q.state_dict(), "v": v.state_dict(), "s_mean": sm, "s_std": ss,
                "KS": KS, "eps": eps, "sd": sd, "arch": "sa_scalar",
                "policy_q": a.policy_q, "omega": a.omega,
                "gamma": a.gamma, "tau": a.tau}, a.out)
    print(f"[IQLak] saved {a.out.split('/')[-1]}  pick%={dist}  mean_k={mean_k:.2f}")
    print("[IQLak] " + json.dumps({"pick": dist, "mean_k": round(mean_k, 3), "eps": eps,
                                   "omega": a.omega, "policy_q": a.policy_q}))


if __name__ == "__main__":
    main()
