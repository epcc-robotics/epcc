"""SAIL row for Table I: initial seed only (r1 = seed 42, the 50 inits every other row uses), best k.
Scoring identical to rescore_mainbench_timematched.py (time-matched hold, park_metrics first/complete,
cube_ttc, median TTC over successes, TP = SAIL Eq.2). All per-env-AWE k on disk: 1.5, 2.0, 2.5, 3.0."""
import sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/baseline"))
import park_metrics as pm  # noqa
B = REPO / "findings/260521_baseline_setup"
TARGET_S = 5.0
SAIL = lambda t, k, rs: [B / f"logs_sail_reached/{t}_sail_awe_per_env_k{k}_n50_r{r}" for r in rs]

def read(run, cube=False, timematched=True):
    runs = run if isinstance(run, (list, tuple)) else [run]
    n_tot, n_ok, ttcs, tmax_s, hks = 0, 0, [], np.nan, []
    for r in runs:
        if not (r / "results.json").is_file():
            print(f"  MISSING results.json: {r}"); continue
        if cube:
            m = pm.cube_ttc(r); hk = None
        else:
            m1 = pm.park_metrics(r, hold_k=1)
            if m1 is None: print(f"  DROPPED: {r}"); continue
            ct = float(m1["results"].get("control_ts_mean", pm.DT))
            hk = max(1, int(round(TARGET_S / ct))) if timematched else 50
            m = pm.park_metrics(r, hold_k=hk)
        if m is None: print(f"  DROPPED: {r}"); continue
        if hk is not None: hks.append(hk)
        ok = np.asarray(m["success"], bool)
        n_tot += int(ok.size); n_ok += int(ok.sum())
        ttcs += list(np.asarray(m["ttc_s"], float)[ok])
        T = np.load(r / "rollout_traj.npz")["action"].shape[0]
        ct_r = .05 if cube else float(m["results"].get("control_ts_mean", pm.DT))
        tmax_s = T * ct_r
    if n_tot == 0: return None
    tpr = (sum(1.0 / max(t, 1e-9) for t in ttcs) - (n_tot - n_ok) / tmax_s) / n_tot
    return dict(sr=n_ok / n_tot, ttc=float(np.median(ttcs)) if ttcs else np.nan, tpr=tpr, n=n_tot, hk=sorted(set(hks)))

TASKS = [("PushT", "plain", False), ("Cube", "cube", True), ("Slippery", "slippery", False)]
KS = ("1.5", "2.0", "2.5", "3.0")
fmt = lambda r: "--" if r is None else f"{100*r['sr']:3.0f} / {r['ttc']:5.1f} / {r['tpr']:+.4f}"
print("TIME-MATCHED 5.0 s hold, initial seed only (r1 = seed 42, n=50)      SR% / TTC s / TP 1/s")
print(f"{'k':>4s}  " + "  ".join(f"{t:>22s}" for t, _, _ in TASKS))
res = {}
for k in KS:
    row = []
    for t, tt, cube in TASKS:
        r = read(SAIL(tt, k, (1,)), cube=cube); res[(t, k)] = r; row.append(f"{fmt(r):>22s}")
    print(f"{k:>4s}  " + "  ".join(row))
print("\n(reference) pooled r1+r2+r3, same scoring")
for k in KS:
    row = [f"{fmt(read(SAIL(tt, k, (1,2,3)), cube=cube)):>22s}" for t, tt, cube in TASKS]
    print(f"{k:>4s}  " + "  ".join(row))
print("\nbest k per task, initial seed:")
for t, _, _ in TASKS:
    cand = {k: res[(t, k)] for k in KS if res[(t, k)] is not None}
    ktp = max(cand, key=lambda k: cand[k]["tpr"]); ksr = max(cand, key=lambda k: (cand[k]["sr"], cand[k]["tpr"]))
    print(f"  {t:9s} by TP: k={ktp}  {fmt(cand[ktp])}    | by SR (tie->TP): k={ksr}  {fmt(cand[ksr])}")
print("DONE")
