"""Table I at equal simulated time per task -> ../scores/table1_eqtime.json.

Rows are the paper's Table I runs (same dirs as the published scoring), plus SuP robot-only
(findings/261023_sup_robot) as an alternative SuP row. For every row two numbers are computed:
  own  - at the run's own length, with the published t_max convention (regression: must equal
         the published value; the job fails otherwise)
  eq   - truncated to T_TASK, t_max = T_TASK
Cube runs are additionally recomputed from the trajectory at full length and compared with the
eval's own success_full / TTC env by env.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eqtime_common import (PUB, REPO, ROOT, T_TASK, cube_metrics, cube_reference, cut,  # noqa: E402
                           push_metrics, summarize)

F = PUB / "findings"
B = "260521_baseline_setup/"
C = "260907_final_dtg_s1_warm/runs/table1/"
RUNS = {
    "pusht": {"DP": B + "logs_sh_repro_wallsoff/pusht_dp_wallsoff",
              "SkiP": B + "logs_sh_repro_wallsoff/pusht_skip_wallsoff",
              "SAIL": B + "logs_sail_reached/plain_sail_awe_per_env_k2.5_n50_r1",
              "SuP": B + "logs_sh_repro_wallsoff/pusht_sup_paperhp",
              "SuP_robot": "261023_sup_robot/runs/pusht_sup_robot_paperhp",
              "PlayGoal": C + "pg_plain_goalsettle",
              "EPCC": C + "t1_plain_M2_s1warm"},
    "cube": {"DP": B + "logs_sail_savetraj/cube_dp_savetraj_n50",
             "SkiP": B + "logs_gate_rollouts/cube_skip_traj_n50",
             "SAIL": B + "logs_sail_reached/cube_sail_awe_per_env_k3.0_n50_r1",
             "SuP": B + "logs_sh_repro_wallsoff/cube_sup_paperhp",
             "SuP_robot": "261023_sup_robot/runs/cube_sup_robot_paperhp",
             "PlayGoal": C + "pg_cube_goalsettle",
             "EPCC": C + "t1_cube_M2_s1warm"},
    "slippery": {"DP": B + "logs_sh_repro_wallsoff/slippery_pusht_dp_wallsoff",
                 "SkiP": B + "logs_sh_repro_wallsoff/slippery_pusht_skip_wallsoff",
                 "SAIL": B + "logs_sail_reached/slippery_sail_awe_per_env_k2.0_n50_r1",
                 "SuP": B + "logs_sh_repro_wallsoff/slippery_sup_paperhp_rel",
                 "SuP_robot": "261023_sup_robot/runs/slippery_sup_robot_paperhp_rel",
                 "PlayGoal": C + "pg_slip_goalsettle",
                 "EPCC": C + "t1_slip_M1_s1warm"},
}
# Published references. Non-SAIL: findings/260907_final_dtg_s1_warm/scores/table1_scores.json (key, run basename).
T1 = json.loads((F / "260907_final_dtg_s1_warm/scores/table1_scores.json").read_text())["rows"]
T1KEY = {"DP": "01_DP", "SkiP": "02_SkiP", "SuP": "05_SpeedupPatch", "PlayGoal": "06_PlayGoal_goalimage"}
EPCCKEY = {"pusht": "07_EPCC", "cube": "07_EPCC", "slippery": "08_EPCC_altWM"}
# SAIL row: logs_smoke/sail_bestk_13246289.out (best k by TP, r1), printed to 1 decimal / 4 decimals
SAIL_REF = {"pusht": (70.0, 19.9, 0.0366), "cube": (96.0, 3.8, 0.2414), "slippery": (48.0, 19.0, 0.0179)}
SUPR = json.loads((F / "261023_sup_robot/scores/table1_sup.json").read_text())


def ref(task, row):
    if row == "SAIL":
        sr, ttc, tp = SAIL_REF[task]
        return dict(SR_pct=sr, TTC_median_s=ttc, TP=tp, tol=dict(SR=1e-6, TTC=0.05 + 1e-9, TP=5e-5 + 1e-9),
                    src="logs_smoke/sail_bestk_13246289.out")
    if row == "SuP_robot":
        r = SUPR[f"{task}/robot"]
        return dict(SR_pct=r["SR_pct"], TTC_median_s=r["TTC_median_s"], TP=r["TP"],
                    tol=dict(SR=1e-6, TTC=1e-4, TP=1e-5), src="261023_sup_robot/scores/table1_sup.json")
    key = f"{task}/{EPCCKEY[task] if row == 'EPCC' else T1KEY[row]}"
    r = T1[key]
    src = r["episodes"][0]["source"]
    assert Path(src).name == Path(RUNS[task][row]).name, (key, src, RUNS[task][row])
    return dict(SR_pct=r["SR_pct"], TTC_median_s=r["TTC_median_s"], TP=r["TP"],
                tol=dict(SR=1e-6, TTC=1e-4, TP=2e-6), src=f"260907 table1_scores.json {key}")


def close(a, b, tol):
    if (a is None or (isinstance(a, float) and math.isnan(a))) and (b is None or (isinstance(b, float) and math.isnan(b))):
        return True
    return abs(float(a) - float(b)) <= tol


out = {"T_task_s": T_TASK, "protocol": "see scripts/eqtime_common.py docstring", "rows": {}, "regression": {}}
fails = 0
for task, rows in RUNS.items():
    T = T_TASK[task]
    for row, rel in rows.items():
        run = F / rel
        if task == "cube":
            full = cube_metrics(run)                    # recomputed, full length
            ref_ok, ref_ttc = cube_reference(run)       # the eval's own success_full / TTC
            same_ok = bool(np.array_equal(full["ok"], ref_ok))
            same_ttc = bool(np.allclose(full["ttc"][ref_ok], ref_ttc[ref_ok], atol=1e-6)) if same_ok else False
            out["regression"][f"{task}/{row}/gtr_recompute"] = dict(success_equal=same_ok, ttc_equal=same_ttc)
            if not (same_ok and same_ttc):
                fails += 1
                print(f"  !! GTR recompute mismatch {task}/{row}: success {same_ok} ttc {same_ttc}", flush=True)
            own = summarize(ref_ok, ref_ttc, full["tmax_own"])
            eqm = cube_metrics(run, T=T)
            eq = summarize(eqm["ok"], eqm["ttc"], T)
            per_ok_eq, per_ttc = eqm["ok"], eqm["ttc"]
            per_ok_own = ref_ok
            extra = dict(sim_time_end_s=dict(min=float(full["sim_time_end"].min()),
                                             max=float(full["sim_time_end"].max())),
                         states_kept_min=int(eqm["kept_states"].min()), steps=full["steps"])
            ep, st = full["ep"], full["st"]
        else:
            m = push_metrics(run)
            own = summarize(m["ok"], m["ttc"], m["tmax_own"])
            ok_eq = cut(m["ok"], m["ttc"], T)
            eq = summarize(ok_eq, m["ttc"], T)
            per_ok_own, per_ok_eq, per_ttc = m["ok"], ok_eq, m["ttc"]
            extra = dict(hold_k=m["hk"], control_ts_mean=m["ct"], steps=m["steps"], t_max_own_s=m["tmax_own"])
            ep, st = m["ep"], m["st"]
        r = ref(task, row)
        good = (close(own["SR_pct"], r["SR_pct"], r["tol"]["SR"])
                and close(own["TTC_median_s"], r["TTC_median_s"], r["tol"]["TTC"])
                and close(own["TP"], r["TP"], r["tol"]["TP"]))
        out["regression"][f"{task}/{row}"] = dict(ok=good, own=own, published=r)
        if not good:
            fails += 1
        print(f"[{'OK' if good else '!!'}] {task:8s} {row:9s} own {own['SR_pct']:5.1f} / {own['TTC_median_s']:6.2f} / "
              f"{own['TP']:+.4f}   pub {r['SR_pct']:5.1f} / {r['TTC_median_s']} / {r['TP']:+.4f}   ->  "
              f"eq@{T}s {eq['SR_pct']:5.1f} / {eq['TTC_median_s']:6.2f} / {eq['TP']:+.4f}", flush=True)
        out["rows"][f"{task}/{row}"] = dict(
            run=rel, own=own, eq=eq, lost_by_cut=int(np.sum(per_ok_own & ~per_ok_eq)), **extra,
            episodes=[dict(env=i, episode=int(ep[i]), start=int(st[i]), ok_own=bool(per_ok_own[i]),
                           ok_eq=bool(per_ok_eq[i]),
                           ttc_s=None if not np.isfinite(per_ttc[i]) else round(float(per_ttc[i]), 4))
                      for i in range(len(ep))])
out["regression_failures"] = fails
dst = ROOT / "scores/table1_eqtime.json"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1, default=float) + "\n")
print(f"\nregression failures: {fails}\nwrote {dst}")
sys.exit(1 if fails else 0)
