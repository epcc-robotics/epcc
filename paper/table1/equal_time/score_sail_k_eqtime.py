"""SAIL speed-factor choice at equal simulated time -> ../scores/sail_k_eqtime.json.

The Table I SAIL row is the best k by TP among k in {1.5, 2, 2.5, 3} (r1 = seed 42), chosen at
each run's own length (logs_smoke/sail_bestk_13246289.out). This checks whether the same k is
still best when every k is scored at T_TASK, as in table1_eqtime.json.
  own - each run at its own length, published convention (regression against sail_bestk: SR,
        TTC to 1 decimal, TP to 4 decimals; the job fails otherwise)
  eq  - truncated to T_TASK, t_max = T_TASK
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eqtime_common import (PUB, REPO, ROOT, T_TASK, cube_metrics, cube_reference, cut,  # noqa: E402
                           push_metrics, summarize)

D = PUB / "findings/260521_baseline_setup/logs_sail_reached"
PREFIX = {"pusht": "plain", "cube": "cube", "slippery": "slippery"}
KS = ["1.5", "2.0", "2.5", "3.0"]
TABLE1_K = {"pusht": "2.5", "cube": "3.0", "slippery": "2.0"}
# logs_smoke/sail_bestk_13246289.out, "TIME-MATCHED 5.0 s hold, initial seed only": SR / TTC / TP
REF = {"pusht": {"1.5": (72, 21.1, 0.0287), "2.0": (72, 18.9, 0.0313), "2.5": (70, 19.9, 0.0366), "3.0": (68, 16.1, 0.0365)},
       "cube": {"1.5": (98, 5.7, 0.1667), "2.0": (100, 4.6, 0.2089), "2.5": (98, 4.1, 0.2314), "3.0": (96, 3.8, 0.2414)},
       "slippery": {"1.5": (42, 25.5, 0.0081), "2.0": (48, 19.0, 0.0179), "2.5": (40, 23.4, 0.0097), "3.0": (42, 25.4, 0.0094)}}

out = {"T_task_s": T_TASK, "table1_k": TABLE1_K, "rows": {}, "best": {}}
fails = 0
for task in ("pusht", "cube", "slippery"):
    T = T_TASK[task]
    for k in KS:
        run = D / f"{PREFIX[task]}_sail_awe_per_env_k{k}_n50_r1"
        if task == "cube":
            full = cube_metrics(run)
            ref_ok, ref_ttc = cube_reference(run)
            same = bool(np.array_equal(full["ok"], ref_ok)) and bool(
                np.allclose(full["ttc"][ref_ok], ref_ttc[ref_ok], atol=1e-6))
            if not same:
                fails += 1
                print(f"  !! GTR recompute mismatch {task} k={k}", flush=True)
            own = summarize(ref_ok, ref_ttc, full["tmax_own"])
            eqm = cube_metrics(run, T=T)
            eq = summarize(eqm["ok"], eqm["ttc"], T)
            tmax_own = full["tmax_own"]
        else:
            m = push_metrics(run)
            own = summarize(m["ok"], m["ttc"], m["tmax_own"])
            eq = summarize(cut(m["ok"], m["ttc"], T), m["ttc"], T)
            tmax_own = m["tmax_own"]
        sr, ttc, tp = REF[task][k]
        good = (abs(own["SR_pct"] - sr) < 1e-6 and abs(own["TTC_median_s"] - ttc) <= 0.05 + 1e-9
                and abs(own["TP"] - tp) <= 5e-5 + 1e-9)
        fails += 0 if good else 1
        print(f"[{'OK' if good else '!!'}] {task:8s} k={k}  own {own['SR_pct']:5.1f} / {own['TTC_median_s']:6.2f} / "
              f"{own['TP']:+.4f} (ref {sr} / {ttc} / {tp:+.4f}, t_max {tmax_own:.1f})  ->  eq@{T}s "
              f"{eq['SR_pct']:5.1f} / {eq['TTC_median_s']:6.2f} / {eq['TP']:+.4f}", flush=True)
        out["rows"][f"{task}/k{k}"] = dict(run=str(run.relative_to(PUB)), own=own, eq=eq, t_max_own_s=float(tmax_own),
                                          regression_ok=good)
    for col in ("own", "eq"):
        tps = {k: out["rows"][f"{task}/k{k}"][col]["TP"] for k in KS}
        best = max(tps, key=tps.get)
        out["best"][f"{task}/{col}"] = dict(k=best, TP=tps[best], same_as_table1=best == TABLE1_K[task],
                                            TP_by_k=tps)
    print(f"  best k by TP: own {out['best'][f'{task}/own']['k']}  eq {out['best'][f'{task}/eq']['k']}  "
          f"(Table I k={TABLE1_K[task]})", flush=True)
out["regression_failures"] = fails
dst = ROOT / "scores/sail_k_eqtime.json"
dst.write_text(json.dumps(out, indent=1, default=float) + "\n")
print(f"\nregression failures: {fails}\nwrote {dst}")
sys.exit(1 if fails else 0)
