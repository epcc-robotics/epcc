"""Synthesize the CMDP dataset D' that the SuP scheduler (IQL) is trained on.

For every replan state in a demo set and every candidate rate k, roll the ADM twice —
once on the fine action chunk A, once on the k-merged chunk A^k — and record the safety
cost E(s,k) = max deviation between them (robot channel: effector only; full channel:
max(effector, object)).

CMDP reward (SuP Prop. 4.1):   r' = k          if E(s,k) <= eps      (safe -> reward the speedup)
                               r' = -OMEGA     otherwise             (unsafe -> heavy penalty)

    python scripts/sup/build_iql_dataset.py --task cube --channel full

Writes ``iql_{channel}_dataset.npz`` (S, K, R, SN, DONE, E, eps, KS) into
``$STABLEWM_HOME/<task>/sup/`` for train_iql.py.
NOTE: the chunk length C must match the eval-time chunk of the policy you will gate
(see README "chunk-length mismatch").
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adm_model import ADM  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KS = [1, 2, 3, 4, 5, 6]      # candidate downsample rates
C = 15                        # action-chunk length used to synthesize D'
CLIP = 1.0                    # clip on the accumulated (summed) delta action
OMEGA = 600.0                 # unsafe penalty; >= gamma*Kmax/(1-gamma) per SuP Prop. 4.1

HOME = os.environ["STABLEWM_HOME"]


def asset_dir(task: str) -> Path:
    """SuP assets live per task: $STABLEWM_HOME/<task_dir>/sup/ (contract)."""
    return Path(HOME) / TASKS[task]["home"] / "sup"

# Demo sets supply the STATES at which the scheduler is queried (not the actions it takes).
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
TRACK = {"robot": "A", "full": "B"}      # channel -> ADM track


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--channel", choices=["robot", "full"], required=True,
                    help="robot = faithful SuP (effector only) | full = ours (+object)")
    ap.add_argument("--max_ep", type=int, default=250)
    ap.add_argument("--chunk", type=int, default=C, help="action-chunk length C (match the gated policy)")
    ap.add_argument("--ckpt_dir", default=None, help="default: $STABLEWM_HOME/<task>/sup")
    ap.add_argument("--out_dir", default=None, help="default: $STABLEWM_HOME/<task>/sup")
    # Added 2026-08-01. The TASKS demo paths above are the non-park sets the
    # scheduler was originally built from, but every other baseline in our table
    # (DP / DTG-MPCC / SkiP) is now trained on the PARK sets. --demo repoints the
    # state source without touching the registry; --tag keeps the park and
    # non-park products side by side so the sensitivity can be measured instead
    # of one silently overwriting the other.
    ap.add_argument("--demo", default=None, help="override the demo h5 (e.g. the park set)")
    ap.add_argument("--tag", default="", help="suffix for the output name: iql_{channel}{tag}_dataset.npz")
    # Added 2026-08-26. KS was hard-coded to 1..6, but the rate menu a gated policy can
    # actually deploy is set by the candidate library, and ours is 1..5 (H*k <= 15 over a
    # 16-waypoint contour). Offering the scheduler a k it cannot be given is not neutral:
    # r' = k rewards the largest safe rate, so an unreachable k=6 competes for probability
    # mass with k=5. It also interacts with the l = C//k truncation below -- at C=15 the
    # merged chunk covers l*k of C, which is 100% for k in {1,3,5} but only 80% for
    # k in {4,6}. Rates evaluated on less of the chunk show a smaller deviation E, look
    # safer, and are over-picked: the shipped cube scheduler picks k=4 at 15% and k=3 at
    # 5%, and k=5 at exactly 0% in all eight shipped configs. Raising C shrinks the spread
    # (96-100% at C=75), and --kmax removes the out-of-menu competitor.
    ap.add_argument("--kmax", type=int, default=6,
                    help="largest candidate rate; KS = 1..kmax (match the deployable menu)")
    # Added 2026-08-26. Query states were carved at non-overlapping stride C, so raising C
    # to match a longer eval chunk divides the dataset by the same factor while MULTIPLYING
    # the state dimension (s = [s0, chunk], so sd = len(cols) + C*adim). Going 15 -> 75 on
    # pushT is 32k transitions of dim 37 down to ~5k of dim 157, which is the wrong
    # direction on both axes and there are only 200 episodes to draw on. --stride decouples
    # how often the scheduler is queried from how much chunk it is shown. Overlapping
    # windows are correlated, so this buys coverage, not independent samples; default
    # stays None = C, which reproduces the shipped behaviour exactly.
    ap.add_argument("--stride", type=int, default=None,
                    help="chunk start stride (default: --chunk, i.e. non-overlapping)")
    a = ap.parse_args()
    KS = list(range(1, a.kmax + 1))

    T = dict(TASKS[a.task])
    if a.demo:
        T["demo"] = a.demo
    print(f"[{a.task}/{a.channel}] demo={T['demo']}  C={a.chunk}  stride={a.stride or a.chunk}  KS={KS}")
    cols = T["cols"][a.channel]
    agent = T["agent"]
    obj = None if a.channel == "robot" else T["obj"]
    adim = T["adim"]
    CC = a.chunk

    ckpt_dir = Path(a.ckpt_dir) if a.ckpt_dir else asset_dir(a.task)
    adm_path = ckpt_dir / f"adm_{TRACK[a.channel]}.pt"
    if not adm_path.is_file():
        raise SystemExit(f"ADM checkpoint missing: {adm_path}\n  train it first: "
                         f"python scripts/sup/train_adm.py --task {a.task} --track {TRACK[a.channel]}")
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
        """ADM rollout: predicted states at horizons 1..len(acts) (any-step, no bootstrap)."""
        H = acts.shape[0]
        s0t = torch.tensor((s0 - sm) / ss, dtype=torch.float32, device=DEV)[None]
        aw = torch.tensor((acts - am) / asd, dtype=torch.float32, device=DEV)[None]
        with torch.no_grad():
            o = m.rollout_multi(s0t, aw, list(range(1, H + 1)))
        return np.stack([s0 + (o[h][0] * ss_t).cpu().numpy() for h in range(1, H + 1)], 0)

    ep_ids = np.unique(epi)
    if len(ep_ids) > a.max_ep:
        ep_ids = np.random.RandomState(0).choice(ep_ids, a.max_ep, replace=False)

    per_ep, allE = {}, []
    for e in ep_ids:
        idx = np.where(epi == e)[0]
        SS = a.stride or CC
        nrep = max((len(idx) - 1 - CC) // SS + 1, 0)
        if nrep < 2:
            continue
        reps = []
        for j in range(nrep):
            t = j * SS
            A = act[idx[t:t + CC]]
            if not np.isfinite(A).all():
                break
            s0 = SB[idx[t]]
            fine = roll(s0, A)
            row = {"s0": s0, "A": A.flatten(), "Ek": {}, "sp": {}}
            for k in KS:
                l = CC // k
                if k == 1 or l < 1:
                    row["Ek"][k] = 0.0; row["sp"][k] = fine[CC - 1]; continue
                Ak = np.clip(A[:l * k].reshape(l, k, adim).sum(1), -CLIP, CLIP)   # accumulate
                mg = roll(s0, Ak)
                ix = [min((jj + 1) * k - 1, CC - 1) for jj in range(l)]           # align fine<->merged
                ee = np.linalg.norm(fine[ix, agent] - mg[:, agent], axis=1)
                if obj is not None:
                    ob = np.linalg.norm(fine[ix, obj] - mg[:, obj], axis=1)
                    E = float(np.maximum(ee, ob).max())
                else:
                    E = float(ee.max())
                row["Ek"][k] = E; row["sp"][k] = mg[-1]; allE.append(E)
            reps.append(row)
        if len(reps) >= 2:
            per_ep[e] = reps

    eps = float(np.quantile(allE, 0.5))   # provisional; train_iql re-picks via --eps_q
    S, K, R, SN, DONE, EE = [], [], [], [], [], []
    for reps in per_ep.values():
        for j, row in enumerate(reps):
            last = j == len(reps) - 1
            nxt = reps[j + 1]["A"] if not last else np.zeros_like(row["A"])
            for k in KS:
                S.append(np.concatenate([row["s0"], row["A"]]))
                K.append(k)
                R.append(float(k) if row["Ek"][k] <= eps else -OMEGA)
                SN.append(np.concatenate([row["sp"][k], nxt]))
                DONE.append(float(last)); EE.append(row["Ek"][k])

    out_dir = Path(a.out_dir) if a.out_dir else asset_dir(a.task)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"iql_{a.channel}{a.tag}_dataset.npz"
    np.savez(out, S=np.array(S, np.float32), K=np.array(K, np.int64), R=np.array(R, np.float32),
             SN=np.array(SN, np.float32), DONE=np.array(DONE, np.float32), E=np.array(EE, np.float32),
             eps=eps, KS=np.array(KS), chunk=CC)
    print(f"[{a.task}/{a.channel}] {len(S)} transitions s_dim={len(S[0])} C={CC} eps={eps:.4f} "
          f"safe={np.mean(np.array(R) > 0) * 100:.0f}%  -> {out}")
    Karr, Earr, Rarr = np.array(K), np.array(EE), np.array(R)
    for k in KS:
        mk = Karr == k
        print(f"    k={k}: safe={np.mean(Rarr[mk] > 0) * 100:3.0f}%  meanE={Earr[mk].mean():.3f}")


if __name__ == "__main__":
    main()
