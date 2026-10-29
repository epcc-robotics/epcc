"""General discrete-action IQL trainer. Usage: python train_iql.py --dataset X.npz --out Y.pt [--eps_q .4]"""
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAU, GAMMA, POLYAK, LR, STEPS, BS, OMEGA = 0.7, 0.99, 0.005, 3e-4, 40000, 512, 600.0


def mlp(i, o, h=256): return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))
class QNet(nn.Module):
    def __init__(s, sd, nk): super().__init__(); s.q1 = mlp(sd, nk); s.q2 = mlp(sd, nk)
    def forward(s, x): return s.q1(x), s.q2(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--eps_q", type=float, default=0.4); ap.add_argument("--omega", type=float, default=OMEGA)
    # Added 2026-08-01. eps is a quantile over the WHOLE deviation distribution,
    # so training on the park generation drags it down: the park tail is free
    # retreat motion with almost no deviation, and including it re-labels
    # contact-phase states as unsafe. Measured: robot channel 1.973 -> 1.313 on
    # pusht, and the scheduler then picks k=1 for 99% of states, i.e. it stops
    # accelerating at all. --eps_from takes the threshold from a different D'
    # (the non-park one) while still TRAINING on this dataset, which separates
    # "which states the scheduler sees" from "what counts as safe".
    ap.add_argument("--eps_from", default=None,
                    help="npz to take the eps quantile from (default: --dataset itself)")
    # Added 2026-08-04. --eps_q and --eps_from are both RELATIVE: eps is a
    # quantile of E, so a fixed fraction of the k>1 rows is labelled unsafe by
    # construction no matter how small the deviations actually are. The SuP
    # paper does not do this -- its eps is an ABSOLUTE deviation threshold,
    # swept in Sec. 5.4 / Tab. 5 over {0.01, 0.015, 0.02} and chosen by success
    # rate. --eps_abs runs the paper's protocol. Units are the ADM's state
    # units and differ per task: px for pusht/slippery (512-px arena), metres
    # for cube -- see the `unit=` field in train_adm.py's task registry. A
    # single number therefore CANNOT be shared across tasks, unlike the paper's
    # apparently global 0.015.
    ap.add_argument("--eps_abs", type=float, default=None,
                    help="absolute deviation threshold (paper Tab.5 style); overrides --eps_q/--eps_from")
    a = ap.parse_args()
    z = np.load(a.dataset)
    S, K, SN, DONE, E = z["S"], z["K"], z["SN"], z["DONE"], z["E"]
    KS = list(z["KS"]); nk = len(KS); k2i = {k: i for i, k in enumerate(KS)}
    if a.eps_abs is not None:
        eps = float(a.eps_abs)
        # Report the quantile this absolute value lands at, because that is the
        # number the relative recipe would have chosen and the gap between them
        # IS the finding. Also print per-k safe% so the label balance the
        # scheduler will see is visible before 40k training steps are spent.
        q = float((E[K > 1] <= eps).mean())
        print(f"[IQL] eps ABSOLUTE {eps:.4f} -> lands at D' quantile {q:.3f} "
              f"(--eps_q {a.eps_q} would have given {float(np.quantile(E[K > 1], a.eps_q)):.4f})")
        for k in KS:
            mk = K == k
            print(f"    k={int(k)}: safe={float((E[mk] <= eps).mean()) * 100:3.0f}%  meanE={float(E[mk].mean()):.4f}")
    elif a.eps_from:
        ze = np.load(a.eps_from)
        eps = float(np.quantile(ze["E"][ze["K"] > 1], a.eps_q))
        print(f"[IQL] eps taken from {a.eps_from.split('/')[-1]} -> {eps:.4f} "
              f"(this dataset would give {float(np.quantile(E[K > 1], a.eps_q)):.4f})")
    else:
        eps = float(np.quantile(E[K > 1], a.eps_q))
    R = np.where(E <= eps, K.astype(np.float32), -a.omega).astype(np.float32)
    sm, ss = S.mean(0), S.std(0) + 1e-6
    Sn = torch.tensor((S - sm) / ss, device=DEV); SNn = torch.tensor((SN - sm) / ss, device=DEV)
    Ai = torch.tensor([k2i[int(k)] for k in K], device=DEV); Rt = torch.tensor(R, device=DEV); Dt = torch.tensor(DONE, device=DEV)
    N, sd = len(S), S.shape[1]
    print(f"[IQL] {a.dataset.split('/')[-1]} N={N} sd={sd} eps={eps:.3f} safe={np.mean(R>0)*100:.0f}%")
    q, qt, v = QNet(sd, nk).to(DEV), QNet(sd, nk).to(DEV), mlp(sd, 1).to(DEV); qt.load_state_dict(q.state_dict())
    oq, ov = torch.optim.Adam(q.parameters(), LR), torch.optim.Adam(v.parameters(), LR)
    for step in range(STEPS):
        ix = torch.randint(0, N, (BS,), device=DEV); s, ac, r, sn, dn = Sn[ix], Ai[ix], Rt[ix], SNn[ix], Dt[ix]
        with torch.no_grad():
            q1t, q2t = qt(s); qsa = torch.minimum(q1t, q2t).gather(1, ac[:, None]).squeeze(1)
        vs = v(s).squeeze(1); vl = (torch.where(qsa - vs >= 0, TAU, 1 - TAU) * (qsa - vs) ** 2).mean()
        ov.zero_grad(); vl.backward(); ov.step()
        with torch.no_grad(): tgt = r + GAMMA * (1 - dn) * v(sn).squeeze(1)
        q1, q2 = q(s); ql = ((q1.gather(1, ac[:, None]).squeeze(1) - tgt) ** 2 + (q2.gather(1, ac[:, None]).squeeze(1) - tgt) ** 2).mean()
        oq.zero_grad(); ql.backward(); oq.step()
        with torch.no_grad():
            for p, pt in zip(q.parameters(), qt.parameters()): pt.data.mul_(1 - POLYAK).add_(POLYAK * p.data)
    with torch.no_grad():
        q1, q2 = q(Sn[:20000]); pick = torch.minimum(q1, q2).argmax(1).cpu().numpy()
    dist = {int(KS[i]): round(int((pick == i).sum()) / len(pick) * 100) for i in range(nk)}
    torch.save({"q": q.state_dict(), "v": v.state_dict(), "s_mean": sm, "s_std": ss, "KS": KS, "eps": eps, "sd": sd}, a.out)
    print(f"[IQL] saved {a.out.split('/')[-1]}  pick%={dist}")


if __name__ == "__main__":
    main()
