#!/usr/bin/env python3
"""Table I under a TIME-MATCHED hold, alongside the published flat hold-50.

The published table scores every pushT row at a flat `hold_k=50`. That is 5.0 s
for the 0.1 s-control methods but only ~2.8 s for SAIL, which runs a faster
control clock -- so SAIL is asked to hold the goal for barely half as long as
everyone else. Fig. 3a already corrects this (plot_friction_plain3way.py derives
hold_k = round(5.0 / control_ts) per run); this script applies the same
correction to the benchmark table so the two agree.

Cube is unaffected: `cube_ttc` scores grasp AND no-drop AND reach, with no hold
window, so there is nothing to time-match. It is recomputed here anyway as a
control -- the numbers must come back identical.

Pooling, TP (SAIL Eq. 2) and t_max follow plot_agg_mainbench_plain_hold50.read()
exactly, so the hold-50 column reproduces the published table bit-for-bit and
the only difference between the two columns is hold_k.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/baseline"))
import park_metrics as pm  # noqa

B = REPO / "findings/260521_baseline_setup"
FIN = REPO / "findings/260907_final_dtg_s1_warm"
TARGET_S = 5.0

SAIL3 = lambda t, k: [B / f"logs_sail_reached/{t}_sail_awe_per_env_k{k}_n50_r{r}" for r in (1, 2, 3)]

ROWS = [
    ("DP",           B / "logs_sh_repro_wallsoff/pusht_dp_wallsoff",
                     B / "logs_sail_savetraj/cube_dp_savetraj_n50",
                     B / "logs_sh_repro_wallsoff/slippery_pusht_dp_wallsoff"),
    ("SkiP",         B / "logs_sh_repro_wallsoff/pusht_skip_wallsoff",
                     B / "logs_gate_rollouts/cube_skip_traj_n50",
                     B / "logs_sh_repro_wallsoff/slippery_pusht_skip_wallsoff"),
    ("SAIL k=2",     SAIL3("plain", "2.0"), SAIL3("cube", "2.0"), SAIL3("slippery", "2.0")),
    ("SAIL k=3",     SAIL3("plain", "3.0"), SAIL3("cube", "3.0"), SAIL3("slippery", "3.0")),
    ("SpeedupPatch", B / "logs_sh_repro_wallsoff/pusht_sup_paperhp",
                     B / "logs_sh_repro_wallsoff/cube_sup_paperhp",
                     B / "logs_sh_repro_wallsoff/slippery_sup_paperhp_rel"),
    ("Play+Goal",    B / "logs_wmfull/wmfull_plain_matched",
                     B / "logs_wmfull/wmfull_cube_matchedM2",
                     B / "logs_wmfull/wmfull_slip_matched"),
    ("EPCC (old)",   B / "logs_fixstats/fix_plainM2_current",
                     B / "logs_agg_eval/agg_cubeM2_guards_w2nu143_fixstats",
                     B / "logs_epcc5/rfx_slipM1_nu15asm1"),
    ("EPCC (ours)",  FIN / "runs/table1/t1_plain_M2_s1warm",
                     FIN / "runs/table1/t1_cube_M2_s1warm",
                     FIN / "runs/table1/t1_slip_M2_s1warm"),
]


def read(run, cube=False, timematched=False):
    """Pooled SR / TTC / TP. Mirrors plot_agg_mainbench_plain_hold50.read()."""
    runs = run if isinstance(run, (list, tuple)) else [run]
    n_tot, n_ok, ttcs, tmax_s, hks = 0, 0, [], np.nan, []
    for r in runs:
        if not (r / "results.json").is_file():
            print(f"  MISSING results.json: {r}")
            continue
        if cube:
            m = pm.cube_ttc(r)
            hk = None
        else:
            m1 = pm.park_metrics(r, hold_k=1)
            if m1 is None:
                print(f"  DROPPED (no rollout_traj): {r}")
                continue
            ct = float(m1["results"].get("control_ts_mean", pm.DT))
            hk = max(1, int(round(TARGET_S / ct))) if timematched else 50
            m = pm.park_metrics(r, hold_k=hk)
        if m is None:
            print(f"  DROPPED (no rollout_traj): {r}")
            continue
        if hk is not None:
            hks.append(hk)
        ok = np.asarray(m["success"], bool)
        n_tot += int(ok.size)          # cube_ttc has no "n" key; park_metrics does
        n_ok += int(ok.sum())
        ttcs += list(np.asarray(m["ttc_s"], float)[ok])
        T = np.load(r / "rollout_traj.npz")["action"].shape[0]
        ct_r = .05 if cube else float(m["results"].get("control_ts_mean", pm.DT))
        tmax_s = T * ct_r
    if n_tot == 0:
        return None
    tpr = (sum(1.0 / max(t, 1e-9) for t in ttcs) - (n_tot - n_ok) / tmax_s) / n_tot
    return dict(sr=n_ok / n_tot, ttc=float(np.median(ttcs)) if ttcs else np.nan,
                tpr=tpr, n=n_tot, hk=sorted(set(hks)))


TASKS = [("PushT", 1, False), ("Cube", 2, True), ("Slippery", 3, False)]

for label, tm in (("FLAT hold_k=50  (published)", False),
                  ("TIME-MATCHED 5.0 s", True)):
    print(f"\n{'='*78}\n{label}\n{'='*78}")
    print(f"{'Method':14s} " + "  ".join(f"{t+' (SR/TTC/TP)':>32s}" for t, _, _ in TASKS))
    for row in ROWS:
        cells = []
        for _, i, cube in TASKS:
            r = read(row[i], cube=cube, timematched=tm)
            if r is None:
                cells.append(f"{'--':>32s}"); continue
            hk = "" if cube else f" hk{'/'.join(map(str, r['hk']))}"
            cells.append(f"{100*r['sr']:5.0f} /{r['ttc']:6.2f} / {r['tpr']:+.4f} n{r['n']:<3d}{hk:>7s}")
        print(f"{row[0]:14s} " + "  ".join(cells))
print("\nDONE")
