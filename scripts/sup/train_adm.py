"""Train the any-step dynamics model (ADM) used by the SuP safety gate.

One trainer for all tasks — pick the corpus/state layout with ``--task``:

    python scripts/sup/train_adm.py --task slippery --track A --epochs 25
    python scripts/sup/train_adm.py --task cube     --track B --epochs 35

Two tracks, same recipe, differing only in the state vector:
  A (SuP-faithful, robot-only) : robot/effector proprio
  B (ours, object-aware)       : A + object (block) dims

Any-step supervision (SuP Eq.6): ONE GRU pass over an ``LMAX`` window supervises every
horizon k=1..LMAX at once (predict Δstate at each prefix). Predictions are never fed back,
so error does not compound with k — that is what makes E(s,k) usable for large k.

Held-out split = last 15% of episode ids. Writes ``adm_{track}.pt`` (weights + normalization
stats + cols) and a ``*_val.json`` (per-horizon error curve) into the task asset dir
``$STABLEWM_HOME/<task>/sup/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adm_model import ADM  # noqa: E402

LMAX = 15
HIDDEN = 512
DEV = "cuda" if torch.cuda.is_available() else "cpu"

HOME = os.environ["STABLEWM_HOME"]


def asset_dir(task: str) -> Path:
    """SuP assets live per task: $STABLEWM_HOME/<task_dir>/sup/ (contract)."""
    return Path(HOME) / TASKS[task]["home"] / "sup"

# Per-task corpus + state layout. `cols` index into the task's raw state/observation array.
#   agent : slice of the TRACK vector holding robot/effector position (error metric)
#   obj   : slice of the TRACK-B vector holding object position (None for track A)
TASKS = {
    "slippery": dict(
        home="slippery_pusht",
        corpus=f"{HOME}/slippery_pusht/data/slippery_pusht_wm_corpus.h5",
        state_key="state", ep_key="episode_idx", adim=2,
        cols={"A": [0, 1, 5, 6], "B": [0, 1, 5, 6, 2, 3, 4]},   # [ax,ay,vx,vy] (+[bx,by,ba])
        agent=slice(0, 2), obj={"A": None, "B": slice(4, 6)}, unit="px",
    ),
    "pusht": dict(
        home="pusht",
        corpus=f"{HOME}/pusht/data/pusht_expert_train.h5",
        state_key="state", ep_key="episode_idx", adim=2,
        cols={"A": [0, 1, 5, 6], "B": [0, 1, 5, 6, 2, 3, 4]},
        agent=slice(0, 2), obj={"A": None, "B": slice(4, 6)}, unit="px",
    ),
    "cube": dict(
        home="cube",
        corpus=f"{HOME}/cube/data/cube_multiview_WM.h5",
        demo=f"{HOME}/cube/data/cube_multiview_diffusion_c03.h5",
        state_key="observation", ep_key="ep_idx", adim=5,
        cols={"A": [12, 13, 14, 15, 16],                          # eff xyz + yaw sincos
              "B": [12, 13, 14, 15, 16, 19, 20, 21, 26, 27]},     # + block xyz + yaw sincos
        agent=slice(0, 3), obj={"A": None, "B": slice(5, 8)}, unit="m",
    ),
}


def load_corpus(T: dict, src: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    with h5py.File(src, "r") as f:
        st = f[T["state_key"]][:].astype(np.float32)
        act = f["action"][:].astype(np.float32)
        epi = f[T["ep_key"]][:]
        hd = f["human_demo"][:] if "human_demo" in f else None
    return st, act, epi, hd


def build_windows(epi, act, hd=None, expert_only=False):
    """Valid window start t: t..t+LMAX inside one episode with all actions finite.
    Split by episode id (last 15% = val) so train/val never share an episode."""
    N = len(epi)
    fin = np.isfinite(act).all(1)
    ep_ids = np.unique(epi)
    if expert_only:
        if hd is None:
            raise SystemExit("--expert_only: corpus has no 'human_demo' flags")
        ep_ids = ep_ids[hd[ep_ids]]          # split WITHIN expert eps, else last-15% eats them all
    n_val = max(1, int(len(ep_ids) * 0.15))
    val_ids = set(ep_ids[-n_val:].tolist())

    idx = np.where(np.arange(N) + LMAX < N)[0]
    idx = idx[epi[idx] == epi[idx + LMAX]]                       # same episode across the span
    cs = np.concatenate([[0], np.cumsum(~fin)])
    idx = idx[(cs[idx + LMAX] - cs[idx]) == 0]                   # all actions finite in span
    if expert_only:
        idx = idx[hd[epi[idx]]]
    is_val = np.array([epi[t] in val_ids for t in idx])
    return idx[~is_val], idx[is_val]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--track", choices=["A", "B"], required=True, help="A=robot-only (SuP), B=+object (ours)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--demo", action="store_true", help="cube: train on the c03 demo set instead of the WM corpus")
    ap.add_argument("--expert_only", action="store_true", help="slippery: keep only human/expert episodes")
    ap.add_argument("--out_dir", default=None, help="default: $STABLEWM_HOME/<task>/sup")
    ap.add_argument("--tag", default="", help="suffix for the checkpoint name")
    a = ap.parse_args()

    T = TASKS[a.task]
    src = T["demo"] if (a.demo and "demo" in T) else T["corpus"]
    out_dir = Path(a.out_dir) if a.out_dir else asset_dir(a.task)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    st, act, epi, hd = load_corpus(T, src)
    cols = T["cols"][a.track]
    S = st[:, cols]
    sd = S.shape[1]
    adim = T["adim"]
    agent, obj = T["agent"], T["obj"][a.track]

    s_mean, s_std = S.mean(0), S.std(0) + 1e-6
    fm = np.isfinite(act).all(1)
    a_mean, a_std = act[fm].mean(0), act[fm].std(0) + 1e-6
    Sn = torch.tensor((S - s_mean) / s_std, device=DEV)
    An = torch.tensor((act - a_mean) / a_std, device=DEV)

    tr, va = build_windows(epi, act, hd, expert_only=a.expert_only)
    print(f"[{a.task} {a.track}] sd={sd} adim={adim} train={len(tr)} val={len(va)} "
          f"src={Path(src).name} dev={DEV} ({time.time() - t0:.1f}s load)")

    model = ADM(sd, adim, hidden=HIDDEN).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    spe = (len(tr) + a.bs - 1) // a.bs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * spe)
    ks = list(range(1, LMAX + 1))
    ss_t = torch.tensor(s_std, device=DEV)

    def batch(ix):
        """-> s0 (B,sd), action window (B,LMAX,adim), target Δstate at every horizon (B,LMAX,sd)."""
        t = torch.as_tensor(ix, device=DEV)
        off = torch.arange(LMAX, device=DEV)
        s0 = Sn[t]
        return s0, An[t[:, None] + off], Sn[t[:, None] + off + 1] - s0[:, None, :]

    def validate():
        model.eval()
        ag = {k: [] for k in ks}
        ob = {k: [] for k in ks}
        with torch.no_grad():
            for i in range(0, len(va), 8192):
                s0, aw, dtar = batch(va[i:i + 8192])
                out, _ = model.gru(aw, model.s_enc(s0).unsqueeze(0).contiguous())
                pred = model.head(out)
                for kk in ks:                                   # denormalize Δ -> raw units
                    dp, dt = pred[:, kk - 1] * ss_t, dtar[:, kk - 1] * ss_t
                    ag[kk].append(torch.linalg.norm(dp[:, agent] - dt[:, agent], dim=-1).cpu().numpy())
                    if obj is not None:
                        ob[kk].append(torch.linalg.norm(dp[:, obj] - dt[:, obj], dim=-1).cpu().numpy())
        return ({k: float(np.concatenate(ag[k]).mean()) for k in ks},
                {k: (float(np.concatenate(ob[k]).mean()) if ob[k] else None) for k in ks})

    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(len(tr))
        tot = nb = 0
        for i in range(0, len(perm), a.bs):
            s0, aw, dtar = batch(tr[perm[i:i + a.bs]])
            out, _ = model.gru(aw, model.s_enc(s0).unsqueeze(0).contiguous())
            loss = ((model.head(out) - dtar) ** 2).mean()       # all horizons at once (Eq.6)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); nb += 1
        if ep % 3 == 0 or ep == a.epochs - 1:
            ac, oc = validate()
            print(f"  ep{ep} loss={tot / nb:.4f} agent@k1={ac[1]:.4f} @k15={ac[15]:.4f} "
                  f"obj@k1={oc[1]} @k15={oc[15]} [{T['unit']}]")

    ac, oc = validate()
    ckpt = out_dir / f"adm_{a.track}{a.tag}.pt"
    torch.save({"state_dict": model.state_dict(), "sd": sd, "adim": adim, "cols": cols,
                "track": a.track, "task": a.task, "hidden": HIDDEN, "lmax": LMAX,
                "s_mean": s_mean, "s_std": s_std, "a_mean": a_mean, "a_std": a_std,
                "agent_curve": ac, "obj_curve": oc}, ckpt)
    json.dump({"task": a.task, "track": a.track, "unit": T["unit"],
               "agent_err_by_k": ac, "obj_err_by_k": oc,
               "train_windows": int(len(tr)), "val_windows": int(len(va)),
               "minutes": (time.time() - t0) / 60},
              open(ckpt.with_name(ckpt.stem + "_val.json"), "w"), indent=2)
    print(f"[{a.task} {a.track}] saved {ckpt}  agent k1={ac[1]:.4f} k15={ac[15]:.4f}  "
          f"obj k1={oc[1]} k15={oc[15]}  ({(time.time() - t0) / 60:.1f}min)")


if __name__ == "__main__":
    main()
