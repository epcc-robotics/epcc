"""D' synthesis, conditioned on the DOWNSAMPLED chunk A^k (paper-faithful variant).

Why this file exists (2026-08-04). Our shipped `build_iql_dataset.py` writes the
scheduler input as

    S = concat([s0, A])            # A = the ORIGINAL chunk

so all six k-rows of a state share an IDENTICAL input vector, and `train_iql.py`
learns Q(s) -> vector indexed by k. The paper does the opposite (Sec 4.2,
Algorithm 1 line 26):

    "the scheduler maps the action chunk A and skip-length k into a downsampled
     representation A^k. By conditioning the policy on (o, A^k), we can treat
     the resulting s' directly as the subsequent state"
    pi(o,A) = argmax_k Q_phi(o, A^k)

i.e. every candidate k produces a DIFFERENT input, and the network sees the
merged action it would actually execute. That is the difference between a
network that can learn a transferable rule ("a merged action this large is
unsafe") and one that must memorise, per state, which of six output heads is
safe — which is exactly the "excellent aggregate predictor but noisy per state"
symptom recorded in scripts/sup/README.md.

Two representations come out of here, both of width `len(cols) + C*adim`:

    S  = [s0,      A          ]   the CMDP state  (o, A)  — before k is chosen
    SQ = [s0,      A^k padded ]   the state-action (o, A^k)
    SN = [sp[k],   A_next     ]   the next CMDP state     — for V

`A^k` has l = C//k merged actions, so it is zero-padded on the right to the full
C*adim width. The padding is part of the representation: a larger k yields both
different values AND fewer non-zero slots, which is exactly the information the
old encoding threw away.

Also fixed here, against the paper:
  * Interpolation. Eq.5 maxes over ALL n timesteps after interpolating the SHORT
    trajectory UP to length n. The shipped file instead subsamples the fine
    trajectory DOWN to the l knots, so deviation BETWEEN knots is never seen and
    E is systematically under-reported. This file interpolates.
  * CLIP. Eq.2's merge is a plain summation; the shipped file hard-clips the sum
    to +/-1 unconditionally. Here `--clip` defaults to inf and should be set to
    1.0 only for envs that really clip (cube).

Everything else — the demo replay, the WM-vs-WM reference, the reward, OMEGA —
is kept identical to the shipped file so this is a single-variable change.

The original `build_iql_dataset.py` is left untouched; it is a collaborator's
file and the existing checkpoints must stay reproducible.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adm_model import ADM  # noqa: E402

import hdf5plugin  # noqa: F401,E402
import h5py  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KS = [1, 2, 3, 4, 5, 6]
C = 15
OMEGA = 600.0
HOME = os.environ["STABLEWM_HOME"]

TASKS = {
    "cube": dict(home="cube", demo=f"{HOME}/cube/data/cube_multiview_diffusion_c03.h5",
                 state_key="observation", ep_key="ep_idx", adim=5,
                 cols={"robot": [12, 13, 14, 15, 16],
                       "full": [12, 13, 14, 15, 16, 19, 20, 21, 26, 27]},
                 agent=slice(0, 3), obj=slice(5, 8)),
    "pusht": dict(home="pusht", demo=f"{HOME}/pusht/data/expert_slow_025_n200.h5",
                  state_key="state", ep_key="ep_idx", adim=2,
                  cols={"robot": [0, 1, 5, 6], "full": [0, 1, 5, 6, 2, 3, 4]},
                  agent=slice(0, 2), obj=slice(4, 6)),
    "slippery": dict(home="slippery_pusht", demo=f"{HOME}/slippery_pusht/data/expert_slow_050_n200_d035.h5",
                     state_key="state", ep_key="ep_idx", adim=2,
                     cols={"robot": [0, 1, 5, 6], "full": [0, 1, 5, 6, 2, 3, 4]},
                     agent=slice(0, 2), obj=slice(4, 6)),
}
TRACK = {"robot": "A", "full": "B"}


def asset_dir(task: str) -> Path:
    return Path(HOME) / TASKS[task]["home"] / "sup"


def interp_to(traj: np.ndarray, n: int) -> np.ndarray:
    """Stretch an (l, d) trajectory to (n, d) by linear interpolation.

    Paper Sec 3.3: hat-tau^k is tau^k "temporally interpolated to match the
    length of tau". The merged rollout's j-th state lands at fine-step
    (j+1)*k-1, so the knots sit on that grid and everything between is linear.
    """
    l = traj.shape[0]
    if l == n:
        return traj
    src = np.linspace(0.0, 1.0, l)
    dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, traj[:, c]) for c in range(traj.shape[1])], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--channel", choices=["robot", "full"], required=True)
    ap.add_argument("--max_ep", type=int, default=250)
    ap.add_argument("--chunk", type=int, default=C)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--demo", default=None, help="override the demo h5 (e.g. the park set)")
    ap.add_argument("--tag", default="", help="suffix for the output name")
    ap.add_argument("--clip", type=float, default=float("inf"),
                    help="clip on the summed delta. Paper Eq.2 has none; use 1.0 only "
                         "for envs that really clip (cube).")
    ap.add_argument("--stride", type=int, default=0,
                    help="chunk-start stride in frames. 0 = C (non-overlapping, the "
                         "shipped behaviour). Algorithm 1 line 11 is stride 1.")
    a = ap.parse_args()

    T = dict(TASKS[a.task])
    if a.demo:
        T["demo"] = a.demo
    print(f"[{a.task}/{a.channel}] demo={T['demo']}  clip={a.clip}  stride={a.stride or a.chunk}")
    cols = T["cols"][a.channel]
    agent = T["agent"]
    obj = None if a.channel == "robot" else T["obj"]
    adim = T["adim"]
    CC = a.chunk
    STRIDE = a.stride if a.stride > 0 else CC

    ckpt_dir = Path(a.ckpt_dir) if a.ckpt_dir else asset_dir(a.task)
    adm_path = ckpt_dir / f"adm_{TRACK[a.channel]}.pt"
    if not adm_path.is_file():
        raise SystemExit(f"ADM checkpoint missing: {adm_path}")
    d = torch.load(adm_path, map_location=DEV, weights_only=False)
    m = ADM(d["sd"], int(d.get("adim", adim)), hidden=d["hidden"]).to(DEV)
    m.load_state_dict(d["state_dict"]); m.eval()
    sm, ss, am, asd = d["s_mean"], d["s_std"], d["a_mean"], d["a_std"]
    ss_t = torch.tensor(ss, device=DEV)

    with h5py.File(T["demo"], "r") as f:
        st = f[T["state_key"]][:].astype(np.float32)
        act = f["action"][:].astype(np.float32)
        epi = f[T["ep_key"]][:]
    SB = st[:, cols]

    def roll(s0, acts):
        H = acts.shape[0]
        s0t = torch.tensor((s0 - sm) / ss, dtype=torch.float32, device=DEV)[None]
        aw = torch.tensor((acts - am) / asd, dtype=torch.float32, device=DEV)[None]
        with torch.no_grad():
            o = m.rollout_multi(s0t, aw, list(range(1, H + 1)))
        return np.stack([s0 + (o[h][0] * ss_t).cpu().numpy() for h in range(1, H + 1)], 0)

    WIDTH = CC * adim   # padded width of the action part of every input vector

    def pad_chunk(Ak: np.ndarray) -> np.ndarray:
        """(l, adim) merged chunk -> (CC*adim,) zero-padded on the right."""
        flat = Ak.reshape(-1)
        out = np.zeros(WIDTH, dtype=np.float32)
        out[:len(flat)] = flat
        return out

    ep_ids = np.unique(epi)
    if len(ep_ids) > a.max_ep:
        ep_ids = np.random.RandomState(0).choice(ep_ids, a.max_ep, replace=False)

    per_ep, allE = {}, []
    for e in ep_ids:
        idx = np.where(epi == e)[0]
        nrep = max(0, (len(idx) - 1 - CC) // STRIDE + 1)
        if nrep < 2:
            continue
        reps = []
        for j in range(nrep):
            t = j * STRIDE
            A = act[idx[t:t + CC]]
            if A.shape[0] < CC or not np.isfinite(A).all():
                break
            s0 = SB[idx[t]]
            fine = roll(s0, A)                       # tau_t  (CC, sd)
            row = {"s0": s0, "A": A.copy(), "Ek": {}, "sp": {}, "Aq": {}}
            for k in KS:
                l = CC // k
                if k == 1 or l < 1:
                    row["Ek"][k] = 0.0
                    row["sp"][k] = fine[CC - 1]
                    row["Aq"][k] = pad_chunk(A)
                    continue
                Ak = np.clip(A[:l * k].reshape(l, k, adim).sum(1), -a.clip, a.clip)
                mg = roll(s0, Ak)                    # tau_t^k  (l, sd)
                # Paper Eq.5: interpolate the SHORT trajectory up to the length
                # of the fine one and max over all of it. The shipped file
                # compared knots only, which never sees between-knot deviation.
                mg_i = interp_to(mg, CC)             # hat-tau_t^k  (CC, sd)
                ee = np.linalg.norm(fine[:, agent] - mg_i[:, agent], axis=1)
                if obj is not None:
                    ob = np.linalg.norm(fine[:, obj] - mg_i[:, obj], axis=1)
                    E = float(np.maximum(ee, ob).max())
                else:
                    E = float(ee.max())
                row["Ek"][k] = E
                row["sp"][k] = mg[-1]
                row["Aq"][k] = pad_chunk(Ak)
                allE.append(E)
            reps.append(row)
        if len(reps) >= 2:
            per_ep[e] = reps

    eps = float(np.quantile(allE, 0.5))   # provisional; the trainer re-picks
    S, SQ, K, R, SN, DONE, EE = [], [], [], [], [], [], []
    for reps in per_ep.values():
        for j, row in enumerate(reps):
            last = j == len(reps) - 1
            nxt = pad_chunk(reps[j + 1]["A"]) if not last else np.zeros(WIDTH, np.float32)
            s_state = np.concatenate([row["s0"], pad_chunk(row["A"])])      # (o, A)
            for k in KS:
                S.append(s_state)                                            # V input
                SQ.append(np.concatenate([row["s0"], row["Aq"][k]]))         # Q input (o, A^k)
                K.append(k)
                R.append(float(k) if row["Ek"][k] <= eps else -OMEGA)
                SN.append(np.concatenate([row["sp"][k], nxt]))               # V input at s'
                DONE.append(float(last)); EE.append(row["Ek"][k])

    out_dir = Path(a.out_dir) if a.out_dir else asset_dir(a.task)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"iqlak_{a.channel}{a.tag}_dataset.npz"
    np.savez(out,
             S=np.array(S, np.float32), SQ=np.array(SQ, np.float32),
             K=np.array(K, np.int64), R=np.array(R, np.float32),
             SN=np.array(SN, np.float32), DONE=np.array(DONE, np.float32),
             E=np.array(EE, np.float32), eps=eps, KS=np.array(KS), chunk=CC,
             clip=a.clip, stride=STRIDE)
    print(f"[{a.task}/{a.channel}] {len(S)} transitions  s_dim={len(S[0])}  C={CC} "
          f"eps={eps:.4f} safe={np.mean(np.array(R) > 0) * 100:.0f}%  -> {out}")
    Karr, Earr, Rarr = np.array(K), np.array(EE), np.array(R)
    for k in KS:
        mk = Karr == k
        print(f"    k={k}: safe={np.mean(Rarr[mk] > 0) * 100:3.0f}%  meanE={Earr[mk].mean():.4f}")
    # The whole point of this variant: Q inputs must actually differ across k.
    sq = np.array(SQ, np.float32).reshape(-1, len(KS), len(SQ[0]))
    spread = float(np.abs(sq - sq[:, :1]).max())
    print(f"    [check] max |SQ(k) - SQ(k=1)| = {spread:.5f}  (0.0 would mean the "
          f"A^k conditioning is not in the input)")


if __name__ == "__main__":
    main()
