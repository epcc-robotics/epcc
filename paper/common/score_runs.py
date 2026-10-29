"""Score campaign run dirs with the paper yardstick (imports park_metrics; never reimplements it).

pushT/slippery: park_metrics(run, hold_k=50, mode="first", ttc_at="complete")  (walls-off, park generation)
cube:           cube_ttc(run)  (grasped AND no_drop AND reached)
TTC = median over successes; TP = SAIL Eq.2 = (sum_success 1/t_i - n_fail/t_max) / n,
t_max = rollout_traj["action"].shape[0] * ct  (ct: cube 0.05, pushT results control_ts_mean).
Identical to make_table1_index.py / table1_ttc.py / plot_cgrid_sweep.py / epcc5h_vs_table1.py.

  python score_runs.py --out scores.json RUN_DIR [RUN_DIR ...]      (compute node: numpy)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/baseline"))
import park_metrics as pm  # noqa

def task_of(run: Path) -> str:
    meta = json.load(open(run / "meta.json"))
    name = str(meta.get("task", {}).get("task_name", ""))
    env = str(meta.get("task", {}).get("env_name", ""))
    return "cube" if ("cube" in name or "Cube" in env) else "pusht"

def score(run: Path) -> dict:
    if not (run / "results.json").is_file() or not (run / "rollout_traj.npz").is_file():
        return {"run": str(run), "error": "missing results.json or rollout_traj.npz"}
    task = task_of(run)
    if task == "cube":
        m = pm.cube_ttc(run); ct = 0.05
    else:
        m = pm.park_metrics(run, hold_k=50, mode="first", ttc_at="complete")
        if m is None:
            return {"run": str(run), "error": "park_metrics returned None (per_env_success_held / control_ts_trace missing?)"}
        ct = float(m["results"].get("control_ts_mean", pm.DT))
    ok = np.asarray(m["success"], bool)
    t = np.asarray(m["ttc_s"], float)
    T = int(np.load(run / "rollout_traj.npz")["action"].shape[0])
    tmax = T * ct
    n = int(ok.size); n_ok = int(ok.sum())
    ts = t[ok]
    tp = (float(np.sum(1.0 / np.maximum(ts, 1e-9))) - (n - n_ok) / tmax) / n
    meta = json.load(open(run / "meta.json"))
    return {
        "run": str(run), "task": task, "n": n, "n_success": n_ok,
        "SR_pct": round(100.0 * n_ok / n, 1),
        "TTC_median_s": round(float(np.median(ts)), 3) if ts.size else None,
        "TP": round(tp, 4), "t_max_s": round(tmax, 3), "rollout_steps": T, "ct": ct,
        "scorer": "cube_ttc" if task == "cube" else "park_metrics(hold_k=50, mode=first, ttc_at=complete)",
        "meta": {
            "policy": meta.get("policy", {}).get("_target_"),
            "wm": meta.get("task", {}).get("policy_config_group"),
            "dataset_stats": meta.get("dataset", {}).get("stats"),
            "task_dataset_stats": meta.get("task", {}).get("dataset_stats"),
            "dtg": meta.get("diffusion", {}).get("ckpt_path"),
            "ref_stride": meta.get("solver", {}).get("reference", {}).get("stride"),
            "c": meta.get("solver", {}).get("reference", {}).get("c"),
            "nu_clamp": meta.get("solver", {}).get("inner_solver", {}).get("nu_clamp"),
            "action_clamp": meta.get("solver", {}).get("inner_solver", {}).get("action_clamp"),
            "env_clip": meta.get("solver", {}).get("inner_solver", {}).get("env_clip_range"),
            "rh": meta.get("plan_config", {}).get("receding_horizon"),
            "H": meta.get("plan_config", {}).get("horizon"),
            "warm_ring": meta.get("warm_ring"),
        },
        "episodes": {"success": ok.astype(int).tolist(), "ttc_s": [None if not np.isfinite(x) else round(float(x), 3) for x in t]},
    }

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    rows = [score(Path(r)) for r in a.runs]
    for r in rows:
        if "error" in r: print(f"{Path(r['run']).name:34s} ERROR {r['error']}"); continue
        print(f"{Path(r['run']).name:34s} {r['task']:5s} n={r['n']:3d} SR={r['SR_pct']:5.1f} TTC={r['TTC_median_s']} TP={r['TP']:+.4f} t_max={r['t_max_s']} stride={r['meta']['ref_stride']} wm={r['meta']['wm']}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"scorer": "scripts/baseline/park_metrics.py (campaign score_runs.py)", "rows": rows}, open(out, "w"), indent=1)
    print("wrote", out)

if __name__ == "__main__":
    main()
