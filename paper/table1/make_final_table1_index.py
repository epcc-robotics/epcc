# FINAL-TREE VARIANT (2026-09-08) of the snapshot make_table1_index.py: only paths/run maps changed.
# EPCC run dirs come from env EPCC_PUSHT / EPCC_CUBE / EPCC_SLIP (absolute paths); defaults = published rows.
"""Build findings/260827_main_benchmark/ — Table I laid out as task -> baseline symlinks.

The runs live scattered across nine logs_* directories for historical reasons, which makes
"where is the SkiP cube run" a search every time. This mirrors the paper's table onto the
filesystem: one directory per task, one entry per row, numbered in table order.

Symlinks are RELATIVE so the tree survives the findings/ directory being moved or copied.
Re-runnable: it rebuilds the links each time, so run it again once WM_full lands.
"""
import os, subprocess, sys, json
E = lambda k, d: os.environ.get(k, d)   # absolute run dir overrides
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"scripts/baseline"))
import park_metrics as pm  # noqa
SRC = REPO/"findings/260521_baseline_setup"
OUT = REPO/"findings/260907_final_dtg_s1_warm"/"table1"
SMOKE = REPO/"logs_smoke"

SAIL = "logs_sail_reached"
ROWS = [
 ("01_DP",           {"pusht": "logs_sh_repro_wallsoff/pusht_dp_wallsoff",
                      "cube":  "logs_sail_savetraj/cube_dp_savetraj_n50",
                      "slippery": "logs_sh_repro_wallsoff/slippery_pusht_dp_wallsoff"}),
 ("02_SkiP",         {"pusht": "logs_sh_repro_wallsoff/pusht_skip_wallsoff",
                      "cube":  "logs_gate_rollouts/cube_skip_traj_n50",
                      "slippery": "logs_sh_repro_wallsoff/slippery_pusht_skip_wallsoff"}),
 ("03_SAIL_k2",      {"pusht": [f"{SAIL}/plain_sail_awe_per_env_k2.0_n50_r{r}" for r in (1,2,3)],
                      "cube":  [f"{SAIL}/cube_sail_awe_per_env_k2.0_n50_r{r}" for r in (1,2,3)],
                      "slippery": [f"{SAIL}/slippery_sail_awe_per_env_k2.0_n50_r{r}" for r in (1,2,3)]}),
 ("04_SAIL_k3",      {"pusht": [f"{SAIL}/plain_sail_awe_per_env_k3.0_n50_r{r}" for r in (1,2,3)],
                      "cube":  [f"{SAIL}/cube_sail_awe_per_env_k3.0_n50_r{r}" for r in (1,2,3)],
                      "slippery": [f"{SAIL}/slippery_sail_awe_per_env_k3.0_n50_r{r}" for r in (1,2,3)]}),
 ("05_SpeedupPatch", {"pusht": "logs_sh_repro_wallsoff/pusht_sup_paperhp",
                      "cube":  "logs_sh_repro_wallsoff/cube_sup_paperhp",
                      "slippery": "logs_sh_repro_wallsoff/slippery_sup_paperhp_rel"}),
 ("06_WM_full_O",       {"pusht": "logs_wmfull/wmfull_plain_O",
                         "cube":  "logs_wmfull/wmfull_cube_O",
                         "slippery": "logs_wmfull/wmfull_slip_O"}),
 # cube's matched twin was first run on agg_cubeM1 -- the EPCC champion at the time --
 # and the champion later moved to agg_cubeM2 when the candidate pool was widened. The
 # M2 rerun is the one that actually holds dynamics fixed and removes only the contour;
 # the M1 run stays on disk as the middle point of the O -> O+B -> O+B+D series.
 # 2026-09-09: the paper's Play+Goal row is the TRUE goal-image variant
 # (policy.goal_frame_only=true, goal_frame_mode=settle -> the goal latent is the frame where the
 # object first reaches the goal and stays). The cells below are matched to the EPCC rows
 # (same WM + scaler, clamps, CEM, plan cadence, episodes, budget). The published
 # `wmfull_*_matched` runs are a MOVING-WINDOW terminal cost (demo waypoint 20 env-steps
 # ahead), kept as 06b for the record -- see README "Play+Goal, true goal-frame variants".
 ("06_PlayGoal_goalimage", {"pusht": E("PG_PUSHT", "__none__"),
                            "cube":  E("PG_CUBE", "__none__"),
                            "slippery": E("PG_SLIP", "__none__")}),
 ("06b_WM_full_movingwindow", {"pusht": "logs_wmfull/wmfull_plain_matched",
                         "cube":  "logs_wmfull/wmfull_cube_matchedM2",
                         "slippery": "logs_wmfull/wmfull_slip_matched"}),
 ("07_EPCC",         {"pusht": E("EPCC_PUSHT", "logs_fixstats/fix_plainM2_current"),
                      "cube":  E("EPCC_CUBE", "logs_agg_eval/agg_cubeM2_guards_w2nu143_fixstats"),
                      "slippery": E("EPCC_SLIP", "logs_epcc5/rfx_slipM1_nu15asm1")}),
 # the other world model of decision D1 (M1 vs M2), for the record; "__none__" -> PENDING
 ("08_EPCC_altWM",   {"pusht": E("EPCC_PUSHT_ALT", "__none__"),
                      "cube":  E("EPCC_CUBE_ALT", "__none__"),
                      "slippery": E("EPCC_SLIP_ALT", "__none__")}),
]

def score(d, cube):
    d = Path(d)
    if not (d/"results.json").is_file() or not (d/"rollout_traj.npz").is_file(): return None
    m = pm.cube_ttc(d) if cube else pm.park_metrics(d, hold_k=50, mode="first", ttc_at="complete")
    if m is None: return None
    ok = np.asarray(m["success"], bool); t = np.asarray(m["ttc_s"], float)
    ct = .05 if cube else float(m["results"].get("control_ts_mean", pm.DT))
    tm = np.load(d/"rollout_traj.npz")["action"].shape[0]*ct
    return ok, t[ok], tm, t

PER_EP = {}          # (task, row) -> per-episode record, persisted next to the index


def pooled(paths, cube, task=None, row=None):
    ok_a, ts_a, tm_a, raw, eps = [], [], [], [], []
    for p in paths:
        v = score(SRC/p, cube)
        if v is None: continue
        ok_a.append(v[0]); ts_a.append(v[1]); tm_a.append(v[2])
        raw.append(float(json.loads((SRC/p/"results.json").read_text())["success_rate"]))
        eps.append({"source": p, "success": [bool(x) for x in v[0]],
                    "ttc_s": [None if not np.isfinite(x) else round(float(x), 4) for x in v[3]],
                    "t_max_s": round(float(v[2]), 4),
                    "raw_success_rate": raw[-1]})
    if task is not None and eps:
        PER_EP[f"{task}/{row}"] = eps
    if not ok_a: return None
    ok = np.concatenate(ok_a); ts = np.concatenate(ts_a); tm = float(np.mean(tm_a))
    # SR = 0 is a result, not a missing row: TTC is undefined but SR and TP are not
    # (TP collapses to -1/t_max). Returning None here blanked cube's Play+Goal cell,
    # which is exactly the row the section is about.
    if not ts.size:
        return (0.0, float("nan"), -1.0 / tm, int(ok.size), float(np.mean(raw)) * 100)
    return (ok.mean()*100, float(np.median(ts)),
            (sum(1/max(x,1e-9) for x in ts)-(ok.size-ok.sum())/tm)/ok.size, ok.size,
            float(np.mean(raw))*100)

def link(dst: Path, target: Path):
    if dst.is_symlink() or dst.exists():
        dst.unlink()                       # never rm -rf a symlink; unlink the LINK only
    dst.symlink_to(os.path.relpath(target, dst.parent))

OUT.mkdir(parents=True, exist_ok=True)
report = []
for task in ("pusht", "cube", "slippery"):
    td = OUT/task; td.mkdir(exist_ok=True)
    for name, m in ROWS:
        paths = m[task]
        paths = paths if isinstance(paths, list) else [paths]
        live = [p for p in paths if (SRC/p).is_dir()]
        if not live:
            report.append((task, name, "PENDING", None, paths[0])); continue
        if len(paths) == 1:
            link(td/name, SRC/paths[0])
            base = Path(paths[0]).name
            # "{base}_*" also matches sibling variants like {base}_cref_<id>; the suffix
            # after the run name must be the bare job id, or the link points at another cell.
            logs = sorted(f for f in SMOKE.glob(f"{base}_*.out")
                          if f.stem[len(base)+1:].isdigit())
            if logs: link(td/f"{name}.joblog.out", logs[-1])
        else:
            rd = td/name; rd.mkdir(exist_ok=True)
            for p in live:
                link(rd/Path(p).name.split("_n50_")[-1], SRC/p)   # r1 / r2 / r3
        v = pooled(live, task == "cube", task, name)
        report.append((task, name, "OK" if len(live) == len(paths) else f"{len(live)}/{len(paths)}",
                       v, paths[0]))

w = max(len(r[1]) for r in report)
lines = ["# Table I — main benchmark (2026-09 campaign: EPCC = H84/K6/stride-1 generator + warm ring), laid out by task",
         "",
         "Symlinks into `findings/260521_baseline_setup/`, one directory per task, entries",
         "numbered in table order. Relative links, so this tree survives being moved.",
         "Rebuild with `python findings/260907_final_dtg_s1_warm/scripts/make_final_table1_index.py` (EPCC_* env = absolute run dirs).",
         "",
         "Scoring: pushT `park_metrics(hold_k=50, mode=\"first\", ttc_at=\"complete\")`, walls-off;",
         "cube `cube_ttc` (grasped AND no-drop AND reached). TP = SAIL Eq.2. SAIL rows pool the",
         "three repeats at episode level.",
         "",
         "\U0001f534 The SR/TTC/TP columns are scored AFTERWARDS by `park_metrics`. The",
         "`results.json` inside each linked directory carries the eval loop's own flag, which on",
         "pushT is a latched-style criterion (one touch of the goal counts) and reads HIGHER --",
         "up to +18 pp (slippery SpeedupPatch 74 vs 56). Cube is unaffected: there",
         "`results.json[\"success_rate\"]` already IS `success_full`, so the two agree exactly.",
         "The `raw` column below is what you will see if you open results.txt.",
         "",
         f"| task | row | SR % | TTC s | TP | n | raw | source |",
         "|---|---|---|---|---|---|---|---|"]
for task, name, st, v, src in report:
    if v is None:
        lines.append(f"| {task} | {name} | — | — | — | — | — | `{src}` ({st}) |")
    else:
        gap = v[4] - v[0]
        raw = f"{v[4]:.0f}" + (f" ({gap:+.0f})" if abs(gap) >= 0.5 else "")
        ttc = "—" if not np.isfinite(v[1]) else f"{v[1]:.2f}"
        lines.append(f"| {task} | {name} | {v[0]:.0f} | {ttc} | {v[2]:.3f} | {v[3]} | {raw} | `{src}` |")
lines += ["",
          "Each linked directory holds `results.json`, `results.txt`, `meta.json`,",
          "`rollout_traj.npz`. `*.joblog.out` is the job stdout where it was kept — the older",
          "baseline runs predate that and have the run directory only.",
          "",
          "Provenance, the two defects found while confirming these numbers, and the EPCC",
          "configs are in the memory note `project_main_benchmark_table_2026_08_27.md`."]
(OUT/"README.md").write_text("\n".join(lines) + "\n")

# The table numbers existed only as rounded text in the README and were recomputed on every
# run, so "where did 76 come from" was not answerable without re-running. Persist the scoring.
SC = REPO/"findings/260907_final_dtg_s1_warm"/"scores"; SC.mkdir(exist_ok=True)
try:
    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "unknown"
except Exception:
    rev = "unknown"
payload = {
    "criterion": {
        "pushT": "park_metrics(hold_k=50, mode='first', ttc_at='complete')",
        "cube": "cube_ttc  (success_full = grasped AND no_drop AND reached)",
        "TP": "SAIL Eq.2  (sum 1/TTC over successes - n_fail/t_max) / n",
        "SAIL rows": "three repeats pooled at episode level",
    },
    "scorer": "scripts/baseline/park_metrics.py",
    "repo_rev": rev,
    "note": ("raw_success_rate is the eval loop's own flag from results.json. On pushT it is a "
             "latched-style criterion and reads higher; on cube it already equals success_full."),
    "rows": {},
}
for task, name, st, v, src in report:
    if v is None: continue
    payload["rows"][f"{task}/{name}"] = {
        "SR_pct": round(v[0], 4), "TTC_median_s": round(v[1], 4), "TP": round(v[2], 6),
        "n_episodes": int(v[3]), "raw_success_rate_pct": round(v[4], 4),
        "episodes": PER_EP.get(f"{task}/{name}", []),
    }
(SC/"table1_scores.json").write_text(json.dumps(payload, indent=2) + "\n")

with open(SC/"table1_episodes.csv", "w") as f:
    f.write("task,row,source,episode,success,ttc_s\n")
    for key, r in payload["rows"].items():
        t, n = key.split("/")
        for blk in r["episodes"]:
            for i, (okv, tv) in enumerate(zip(blk["success"], blk["ttc_s"])):
                f.write(f"{t},{n},{blk['source']},{i},{int(okv)},{'' if tv is None else tv}\n")
print(f"wrote {SC}/table1_scores.json and table1_episodes.csv")
for r in report:
    print(f"  {r[0]:<9}{r[1]:<{w}}  {r[2]}")
print(f"\nwrote {OUT}/README.md")
