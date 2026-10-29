"""Table I SuP row: published (full channel) vs robot-only, with the paper's scorer.

score() and pooled() are copied from 260907_final_dtg_s1_warm/scripts/make_final_table1_index.py
(that file builds symlinks at import, so it is not imported). The published column must
reproduce the paper (88/13.4/.062 · 98/3.2/.318 · 56/29.3/.017) or the new column is not
trusted. Also prints the last-step criterion of Appendix I (park_metrics mode="final").
Writes ../scores/table1_sup.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/baseline"))
import park_metrics as pm  # noqa: E402

OLD = REPO / "findings/260521_baseline_setup/logs_sh_repro_wallsoff"
NEW = REPO / "findings/261023_sup_robot/runs"
RUNS = {"pusht": ("pusht_sup_paperhp", "pusht_sup_robot_paperhp"),
        "cube": ("cube_sup_paperhp", "cube_sup_robot_paperhp"),
        "slippery": ("slippery_sup_paperhp_rel", "slippery_sup_robot_paperhp_rel")}
REF = json.load(open(REPO / "findings/260907_final_dtg_s1_warm/scores/table1_scores.json"))["rows"]


def score(d, cube):
    d = Path(d)
    if not (d/"results.json").is_file() or not (d/"rollout_traj.npz").is_file(): return None
    m = pm.cube_ttc(d) if cube else pm.park_metrics(d, hold_k=50, mode="first", ttc_at="complete")
    if m is None: return None
    ok = np.asarray(m["success"], bool); t = np.asarray(m["ttc_s"], float)
    ct = .05 if cube else float(m["results"].get("control_ts_mean", pm.DT))
    tm = np.load(d/"rollout_traj.npz")["action"].shape[0]*ct
    return ok, t[ok], tm, t


def pooled(d, cube):
    v = score(d, cube)
    ok, ts, tm = v[0], v[1], float(v[2])
    if not ts.size:
        return (0.0, float("nan"), -1.0 / tm, int(ok.size))
    return (ok.mean()*100, float(np.median(ts)),
            (sum(1/max(x, 1e-9) for x in ts)-(ok.size-ok.sum())/tm)/ok.size, ok.size)


out, bad = {}, 0
print(f"{'task':9s} {'':10s} {'SR%':>5s} {'TTC s':>6s} {'TP':>7s}  n   last-step SR% (hold 50, mode=final)")
for task, (o, n) in RUNS.items():
    cube = task == "cube"
    for tag, d in (("published", OLD / o), ("robot", NEW / n)):
        sr, ttc, tp, nn = pooled(d, cube)
        fin = None
        if not cube:
            mf = pm.park_metrics(d, hold_k=50, mode="final", ttc_at="complete")
            fin = float(np.asarray(mf["success"], bool).mean() * 100)
        out[f"{task}/{tag}"] = dict(run=str(d.relative_to(REPO)), SR_pct=round(sr, 2),
                                    TTC_median_s=round(ttc, 3), TP=round(tp, 6), n=nn,
                                    last_step_SR_pct=fin)
        print(f"{task:9s} {tag:10s} {sr:5.0f} {ttc:6.1f} {tp:7.3f}  {nn}   {fin if fin is not None else '-'}")
        if tag == "published":   # exact match against the published score file (paper rounds half-up)
            ref = REF[f"{task}/05_SpeedupPatch"]
            if (round(sr, 4), round(ttc, 4), round(tp, 6)) != (ref["SR_pct"], ref["TTC_median_s"], ref["TP"]):
                print(f"   🔴 published column does not reproduce table1_scores.json {ref['SR_pct'], ref['TTC_median_s'], ref['TP']}"); bad += 1
(REPO / "findings/261023_sup_robot/scores").mkdir(exist_ok=True)
(REPO / "findings/261023_sup_robot/scores/table1_sup.json").write_text(json.dumps(out, indent=1) + "\n")
print("reproduces paper:", "YES" if not bad else "NO")
sys.exit(1 if bad else 0)
