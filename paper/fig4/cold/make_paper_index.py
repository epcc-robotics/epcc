#!/usr/bin/env python3
"""Build the run indexes of the two paper figures re-derived on 2026-10-22/23, in the same
convention as findings/260907_final_dtg_s1_warm/table1/<task>/NN_<row> (relative symlinks,
one directory per task, numbered in figure order, .joblog.out = eval stdout).

  findings/261022_fig4_cold/fig4/<task>/NN_<point>          Fig. 4 (pace sweep, cold predictor)
      NN_<point>               -> eval run dir (results.json, meta.json, rollout_traj.npz)
      NN_<point>.joblog.out    -> eval job stdout
      NN_<point>.wm_ckpt       -> the world-model checkpoint the eval loaded
      NN_<point>.train_log[.N] -> training job stdout (two files for a resumed run)
  findings/261022_fig3b_sail/fig3b/<task>/NN_<method>      Fig. 3b (phase speed-up, Table I SAIL)

Standard library only (login-node safe). Rerun after any change; it rebuilds both trees.
"""
import json
import os
import re
from pathlib import Path

R = Path(__file__).resolve().parents[3]
F = R / "findings"
C = F / "260907_final_dtg_s1_warm"
B = F / "260521_baseline_setup"
N = F / "261022_fig4_cold"
S3 = F / "261022_fig3b_sail"
L = R / "logs_smoke"
H = Path(os.environ["STABLEWM_HOME"])


def rel_link(link: Path, target: Path):
    """Relative link inside the project tree (survives moving it); absolute link to scratch."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    target = Path(target)
    if str(target).startswith(str(R)):
        link.symlink_to(os.path.relpath(target, link.parent))
    else:
        link.symlink_to(os.path.realpath(target))


def latest_log(prefix):
    logs = sorted(L.glob(f"{prefix}_[0-9]*.out"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    return logs[-1] if logs else None


def all_logs(prefix):
    """Training logs that actually trained (>= 2 validation tables): drops first tries that died at
    start-up (wandb service timeout, 2026-10-22) and smoke tests."""
    out = []
    for p in sorted(L.glob(f"{prefix}_[0-9]*.out"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
        if "smoke" in p.name:
            continue
        if p.read_text(errors="replace").count("validate/pred_loss_epoch") >= 2:
            out.append(p)
    return out


# ------------------------------------------------------------------ Fig. 4 --
PACE = {"pusht": {"D05": 1.85, "CTRL": 3.51, "D15": 4.15, "D25": 5.30},
        "slippery": {"D05": 1.44, "CTRL": 2.47, "D15": 3.28, "D25": 4.21},
        "cube": {"D03": 1.06, "D075": 2.65, "CTRL": 3.54, "D12": 4.22}}
C_OF = {"D05": 0.5, "CTRL": 1.0, "D15": 1.5, "D25": 2.5, "D03": 0.3, "D075": 0.75, "D12": 1.2}
TASKKEY = {"pusht": "Plain PushT", "cube": "Cube", "slippery": "Slippery PushT"}
new_sc = json.load(open(N / "scores/fig4_scores.json"))
old_sc = json.load(open(C / "scores/fig4_scores.json"))


def score(sc, task, point):
    t = sc[TASKKEY[task]]
    if point == "DEMO":
        return t["demo"]
    return dict(t["points"])[C_OF[point]]


rows = []
for task, order in [("pusht", ["DEMO", "D05", "CTRL", "D15", "D25"]),
                    ("cube", ["DEMO", "D03", "D075", "CTRL", "D12"]),
                    ("slippery", ["DEMO", "D05", "CTRL", "D15", "D25"])]:
    for i, p in enumerate(order):
        label = "ExpertOnly" if p == "DEMO" else f"play_{PACE[task][p]:.2f}x_{p}"
        base = N / "fig4" / task / f"{i:02d}_{label}"
        tk = "plain" if task == "pusht" else task
        published_cube = task == "cube" and p in ("DEMO", "CTRL")    # not re-run: already cold
        if published_cube:
            run = C / f"runs/fig_expert_play/f4_cube_{p}_s1warm"
            ev = latest_log(f"f4_cube_{p}_s1warm")
            wm = (H / "checkpoints" / ("wm_demoonly/cube_demo" if p == "DEMO" else "wm_playsub/cube")).with_name(
                ("cube_demo" if p == "DEMO" else "cube") + "_object.ckpt")
            tlogs = [L / "pv_cube_demoonly_11583824.out"] if p == "DEMO" else [L / "pv_cube_playsub_11608139.out"]
            note = "published run (Cube predictor was already cold)"
        else:
            run = N / f"runs/fig_expert_play/f4c_{tk}_{p}"
            ev = latest_log(f"f4c_{tk}_{p}")
            wm = H / f"checkpoints/f4cold/{tk}_{p}_object.ckpt"
            if task == "cube":
                c = {"D03": "c0.3", "D075": "c0.75", "D12": "c1.2"}[p]
                tlogs = [x for x in all_logs(f"wm_agg_cube_D_{c}")]
                note = "re-evaluated at the lowest-validation epoch (no retraining)"
            else:
                tlogs = [x for x in all_logs(f"f4cold_{tk}_{p}")]
                note = "predictor from scratch, 100 epochs, lowest validation prediction loss"
        rel_link(base, run)
        for old in base.parent.glob(base.name + ".train_log*"):
            old.unlink()
        if ev:
            rel_link(base.with_name(base.name + ".joblog.out"), ev)
        rel_link(base.with_name(base.name + ".wm_ckpt"), Path(os.path.realpath(wm)))
        for j, tl in enumerate(tlogs):
            rel_link(base.with_name(base.name + (".train_log" if j == 0 else f".train_log.{j + 1}")), tl)
        s_new, s_old = score(new_sc, task, p), score(old_sc, task, p)
        ep = re.search(r"_epoch_(\d+)_object", os.path.realpath(wm))
        rows.append((task, base.name, s_new, s_old, ep.group(1) if ep else "?", ev.name if ev else "-", note))

fmt = lambda s: f"{100 * s['sr']:.0f} / {s['ttc']:.2f} / {s['tpr']:.3f}"
md = ["# Fig. 4 — expert–play complementarity, cold-predictor rerun (2026-10-22/23) — paper version",
      "",
      "Replaces `findings/260907_final_dtg_s1_warm/runs/fig_expert_play/f4_*` for PushT and Slippery-PushT",
      "(those continued the predictor of the play-trained first-stage world model) and the Cube D points",
      "(deployed at the last epoch instead of the lowest-validation one). Memory:",
      "`project_fig4_cold_rerun_2026_10_22.md`.",
      "",
      "Per point: `NN_<point>` -> eval run dir · `.joblog.out` eval stdout · `.wm_ckpt` the world model that",
      "was loaded · `.train_log[.N]` training stdout (.2 = resumed job). Figure:",
      "`../fig_expert_play/cgrid_sweep_1to1_meas.png`; scores `../scores/fig4_scores.json`; per-point new-vs-old",
      "`../scores/<task><POINT>_new_vs_old.json`; plot `../scripts/plot_cgrid_sweep_1to1.py` with",
      "`FIG_W=3.54 FIG_H=2.60 FIG_PT=8 COMPACT=1 SHOW_PREV=0 RIGHT_RECT=0.955 MEASURED_X=1 SUF=_meas FILLED=1 ONLY=sweep`",
      "(run through `scripts/baseline/run_analysis.sbatch`). Scoring as for Table I (park_metrics hold 50 first/complete;",
      "cube_ttc). x = measured pace (p90 command magnitude / demo p90).",
      "",
      "| task | point | SR % / TTC s / TP (this version) | published | WM epoch | eval log | note |",
      "|---|---|---|---|---|---|---|"]
md += [f"| {t} | {n} | {fmt(a)} | {fmt(b)} | {e} | `{lg}` | {nt} |" for t, n, a, b, e, lg, nt in rows]
md += ["",
       "Sec. V-D numbers from this version: Expert-Only mean SR (4+10+30)/3 = 14.7 %; best play (62+88+38)/3 = 62.7 %;",
       "fastest pace (10+80+10)/3 = 33.3 %; SR peak at 3.51/2.65/3.28x, TP peak at 3.51/3.54/3.28x; Expert-Only TP < 0",
       "on PushT and Cube only.",
       "",
       "Rebuild: `python3 findings/261022_fig4_cold/scripts/make_paper_index.py`"]
(N / "fig4/README.md").write_text("\n".join(md) + "\n")

# ----------------------------------------------------------------- Fig. 3b --
F3 = {
    "pusht": [("01_EPCC", C / "runs/table1/t1_plain_M2_s1warm"),
              ("02_SuP", B / "logs_sh_repro_wallsoff/pusht_sup_paperhp"),
              ("03_SAIL_k2.5_seed42", B / "logs_sail_reached/plain_sail_awe_per_env_k2.5_n50_r1"),
              ("04_SkiP", B / "logs_sh_repro_wallsoff/pusht_skip_wallsoff")],
    "cube": [("01_EPCC", C / "runs/table1/t1_cube_M2_s1warm"),
             ("02_SuP", B / "logs_sh_repro_wallsoff/cube_sup_paperhp"),
             ("03_SAIL_k3.0_seed42", B / "logs_sail_reached/cube_sail_awe_per_env_k3.0_n50_r1"),
             ("04_SkiP", B / "logs_gate_rollouts/cube_skip_traj_n50")],
    "slippery": [("01_EPCC", C / "runs/table1/t1_slip_M1_s1warm"),
                 ("02_SuP", B / "logs_sh_repro_wallsoff/slippery_sup_paperhp_rel"),
                 ("03_SAIL_k2.0_seed42", B / "logs_sail_reached/slippery_sail_awe_per_env_k2.0_n50_r1"),
                 ("04_SkiP", B / "logs_sh_repro_wallsoff/slippery_pusht_skip_wallsoff")],
}
for task, items in F3.items():
    for name, target in items:
        assert target.exists(), target
        rel_link(S3 / "fig3b" / task / name, target)
vals = {p["panel"]: p for p in json.load(open(S3 / "out/tableI/contact_split_3task_precision_values.json"))}
pub = {p["panel"]: p for p in json.load(open(S3 / "out/published/contact_split_3task_precision_values.json"))}
md = ["# Fig. 3b — speed-up in contact vs free motion, with the Table I SAIL rows (2026-10-22) — paper version",
      "",
      "Same inputs as Table I: EPCC = Table I cells, SuP / SkiP = Table I baselines, SAIL = the Table I rows",
      "(seed 42, best-TP k = 2.5 / 3 / 2) scored with the Table I time-matched 5 s hold (hold_k = round(5/ct):",
      "99 PushT, 88 Slippery; 50 for every other method). The published Fig. 3b used SAIL k = 3 with a flat",
      "50-step hold (`out/published/` reproduces it exactly: paired 22/33/8). Cube map = `precision`",
      "(contact = grasp + place).",
      "",
      "Figure `../out/tableI/contact_split_3task_precision_1to1_prec.png`, values",
      "`../out/tableI/contact_split_3task_precision_values.json`, script",
      "`../scripts/contact_split_3task_sailcheck.py` with `SAIL_MODE=tableI` and",
      "`EPCC_PUSHT/CUBE/SLIP = $C/runs/table1/t1_{plain_M2,cube_M2,slip_M1}_s1warm FIG_W=3.54 FIG_H=1.80 FIG_PT=8",
      "PHASE_XLAB='motion phase' RIGHT_RECT=0.94 CUBE_MAP=precision SUF=_prec` via `scripts/baseline/run_analysis.sbatch`.",
      "",
      "| panel | paired episodes | method | contact α | free α | published contact / free |",
      "|---|---|---|---|---|---|"]
for panel in ("PushT", "Cube", "Slippery-PushT"):
    v, q = vals[panel], pub[panel]
    for m in ("EPCC", "SuP", "SAIL", "SkiP"):
        a, b = v["alpha"][m], q["alpha"][m]
        md.append(f"| {panel} | {v['paired_n']} (was {q['paired_n']}) | {m} | {a['contact']:.2f} | "
                  f"{a['free motion']:.2f} | {b['contact']:.2f} / {b['free motion']:.2f} |")
md += ["",
       "Paper numbers from this version: Sec. I contact speed-up over the strongest baseline per task, averaged:",
       "PushT 3.03/2.65, Cube 2.00/1.69, Slippery 1.80/1.47 (SAIL) -> 1.18x (was 1.35x). Sec. V-B means: contact",
       "EPCC 2.28x vs SuP 1.78x (was 2.53 vs 1.91), free 1.70x vs 1.61x (was 1.73 vs 1.67).",
       "",
       "Rebuild: `python3 findings/261022_fig4_cold/scripts/make_paper_index.py`"]
(S3 / "fig3b/README.md").write_text("\n".join(md) + "\n")
print("fig4 rows:", len(rows), " fig3b links:", sum(len(v) for v in F3.values()))
