"""TTC (median over successes, pooled over the 3 SAIL seeds) at the time-matched hold
(hold_k = round(5.0 / control_ts) per run) — the same criterion the manuscript's SAIL row uses."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/baseline"))
import park_metrics as pm
B = Path(__file__).resolve().parents[2] / "findings/260521_baseline_setup"
CELLS = {("PushT","k2"): [B/f"logs_sail_reached/plain_sail_awe_per_env_k2.0_n50_r{i}" for i in (1,2,3)],
         ("PushT","k3"): [B/f"logs_sail_reached/plain_sail_awe_per_env_k3.0_n50_r{i}" for i in (1,2,3)],
         ("Slippery","k2"): [B/f"logs_sail_reached/slippery_sail_awe_per_env_k2.0_n50_r{i}" for i in (1,2,3)],
         ("Slippery","k3"): [B/f"logs_sail_reached/slippery_sail_awe_per_env_k3.0_n50_r{i}" for i in (1,2,3)]}
for key, runs in CELLS.items():
    for label, tm in (("flat hk50", False), ("time-matched 5s", True)):
        ttcs, succ, n, hks = [], 0, 0, []
        for r in runs:
            m1 = pm.park_metrics(r, hold_k=1, mode="first", ttc_at="complete")
            # park_metrics exposes the period under results["control_ts_mean"]; the old
            # lookup used a key it never returns, so every "time-matched" row silently
            # fell back to hold_k=50 and reproduced the flat row.
            ct = float(m1["results"].get("control_ts_mean", 0.1))
            hk = int(round(5.0/ct)) if tm else 50
            hks.append(hk)
            m = pm.park_metrics(r, hold_k=hk, mode="first", ttc_at="complete")
            s = np.asarray(m["success"], bool); t = np.asarray(m["ttc_s"], float)
            ttcs += list(t[s]); succ += int(s.sum()); n += len(s)
        print(f"{key[0]:<9} SAIL {key[1]}  {label:<16} SR {100*succ/n:5.1f}  TTC median {np.median(ttcs):6.2f}s  (n={n}, hk={sorted(set(hks))})")
