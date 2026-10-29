# FINAL-TREE VARIANT (2026-09-08) of the snapshot phase_compare_cube.py: only paths/run maps changed.
# EPCC run dirs come from env EPCC_PUSHT / EPCC_CUBE / EPCC_SLIP (absolute paths); defaults = published rows.
#!/usr/bin/env python3
"""Cube per-phase speedup, on the Table I baselines.

alpha_phase = (demo steps in the phase x 0.05 s) / (method sim-time in the phase)

🔴 The sim-time denominator is SUMMED FROM THE REALISED CONTROL PERIOD, not
(steps x a constant). SAIL-with-AWE switches its control period per step -- the cube
traces hold {0.025, 0.0521} at k=2 and {0.0167, 0.0521} at k=3 -- so the old
`scale = 0.05 / control_ts` was only ever right for the fixed-period methods and
silently mis-scaled every AWE row. Methods with a constant period are unaffected:
summing 0.05 over n steps is the same number.

Rows are the Table I cells (findings/260827_main_benchmark), so this figure and the
main benchmark cannot disagree about which run a method refers to. DP is omitted: at
alpha ~ 1 everywhere it is the demo-pace reference the dashed line already draws.

Supersedes phase_compare_4method.py, which pinned older cells and a constant period.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stable_worldmodel as swm  # noqa
from latent_modules.eval.demo_ttc import compute_demo_ttc_for_run, _episode_state_seq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase_gripper_split import events, P  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/baseline"))
from paper_style import (paper_style, save_fig, series_color, series_hatch,  # noqa: E402
                         sparse_ticks, DOUBLE_COL)

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "findings/260521_baseline_setup"
DS = "cube_multiview_diffusion_c0.3_postpad_fullscale"
TASK = "cube-mv"
CACHE = os.environ.get("STABLEWM_HOME")
DEMO_CT = 0.05
OUT = REPO/"findings/260907_final_dtg_s1_warm" / "fig_phase"
SAILD = BASE / "logs_sail_reached"

# ours is index 0 — the style module keys the foreground colour off that.
METHODS = [
    ("EPCC (ours)",   [Path(os.environ.get("EPCC_CUBE", str(BASE / "logs_agg_eval/agg_cubeM2_guards_w2nu143_fixstats")))]),
    ("SpeedupPatch",  [BASE / "logs_sh_repro_wallsoff/cube_sup_paperhp"]),
    ("SAIL $k{=}2$",  [SAILD / f"cube_sail_awe_per_env_k2.0_n50_r{r}" for r in (1, 2, 3)]),
    ("SAIL $k{=}3$",  [SAILD / f"cube_sail_awe_per_env_k3.0_n50_r{r}" for r in (1, 2, 3)]),
    ("SkiP",          [BASE / "logs_gate_rollouts/cube_skip_traj_n50"]),
]


def _ct_trace(run, T, n_env):
    """Per-step realised control period, (T, n_env). Falls back to a constant 0.05
    for runs recorded before the trace existed (all of which held the period fixed)."""
    f = run / "control_ts_trace.npz"
    if f.is_file():
        z = np.load(f, allow_pickle=True)
        # "requested", not "realised": park_metrics (the single source of truth for
        # Table I) times every row on the requested period, and the figure must not use a
        # different clock than the table. They are identical everywhere except cube SAIL,
        # whose slow branch realises 0.0521 s instead of 0.05 -- at most 0.058 s per
        # episode, which moves that row's alpha by ~1-3% and no ordering.
        k = "requested" if "requested" in z.files else z.files[0]
        a = np.asarray(z[k], float)
        if a.ndim == 2 and a.shape[1] == n_env:
            out = np.full((T, n_env), np.nan)
            m = min(T, a.shape[0])
            out[:m] = a[:m]
            # a step past the trace keeps the last observed period
            for j in range(n_env):
                last = a[m - 1, j] if m else DEMO_CT
                out[m:, j] = last
            return np.nan_to_num(out, nan=DEMO_CT)
    return np.full((T, n_env), DEMO_CT)


def per_phase_alpha(runs):
    """Per-episode alphas, keyed by the DEMO episode index so methods can be paired.

    Env index is not a usable key: the repeats draw different episode sets (r1 starts at
    ep 312, r2 at ep 210, r3 at ep 564), so env 0 is a different problem in each.
    """
    per = {}                      # ep -> {phase: alpha}
    ds = swm.data.HDF5Dataset(DS, keys_to_cache=[], cache_dir=CACHE)
    seqc = {}
    for run in runs:
        z = np.load(run / "rollout_traj.npz")
        epi = np.asarray(z["episode_idx"], np.int64)
        s0 = np.asarray(z["start_step"], np.int64)
        succ = np.asarray(z["success_full"]).astype(bool)
        fss = np.asarray(z["first_success_step"], float)
        eff, cube, goal = z["effector_pos"], z["cube_pos"], z["goal_pos"]
        tzc = np.asarray(z["table_z"], float)
        grip = z["gripper"]
        T, n_env = eff.shape[0], eff.shape[1]
        ct = _ct_trace(run, T, n_env)

        import yaml
        yaml.safe_dump(dict(task=TASK, eval_dataset=DS,
                            episode_idx=[int(x) for x in epi],
                            start_step=[int(x) for x in s0], cache_dir=str(CACHE)),
                       open(run / "meta.yaml", "w"))
        demo_ttc = compute_demo_ttc_for_run(run, criterion="gtr", force=True, cache_dir=CACHE)

        for i in range(len(epi)):
            if not (succ[i] and fss[i] > 3 and demo_ttc[i] > 3):
                continue
            ep = int(epi[i])
            if ep not in seqc:
                seqc[ep] = _episode_state_seq(ds.load_episode(ep), TASK)
            seq = seqc[ep]
            a, D = int(s0[i]), int(demo_ttc[i])
            win = seq[a:a + D + 1]
            if len(win) < 4 or "gripper_opening" not in win[0]:
                continue
            de = np.array([np.asarray(s["effector_pos"]) for s in win])
            dc = np.array([np.asarray(s["cube_pos"]) for s in win])
            dgr = np.array([float(s["gripper_opening"]) for s in win])
            dg = np.asarray(win[0].get("goal_pos", goal[i]))
            dtz = float(win[0].get("table_z", dc[0, 2]))
            ed = events(de, dc, dg, dtz, dgr, D)
            eo = events(eff[:, i, :], cube[:, i, :], goal[i], tzc[i], grip[:, i], fss[i])
            keys = ("t1", "tc", "tl", "t4", "t5")
            if any(ed[k] <= 0 for k in keys) or any(eo[k] <= 0 for k in keys):
                continue
            # (start, end) step indices per phase, demo side and method side
            db = dict(above=(0, ed["t1"]), descend=(ed["t1"], ed["tc"]),
                      grasp=(ed["tc"], ed["tl"]), lift=(ed["tl"], ed["t4"]),
                      carry=(ed["t4"], ed["t5"]), place=(ed["t5"], ed["t6"]))
            ob = dict(above=(0, eo["t1"]), descend=(eo["t1"], eo["tc"]),
                      grasp=(eo["tc"], eo["tl"]), lift=(eo["tl"], eo["t4"]),
                      carry=(eo["t4"], eo["t5"]), place=(eo["t5"], eo["t6"]))
            if any(db[p][1] - db[p][0] <= 0 or ob[p][1] - ob[p][0] <= 0 for p in P):
                continue
            row = {}
            for p in P:
                d_s = (db[p][1] - db[p][0]) * DEMO_CT               # demo sim-time
                lo, hi = int(ob[p][0]), min(int(ob[p][1]), T)       # method sim-time
                o_s = float(ct[lo:hi, i].sum())
                if o_s <= 0:
                    row = None
                    break
                row[p] = d_s / o_s
            if row is not None:
                per.setdefault(ep, []).append(row)
    # an episode seen in more than one repeat contributes its median, once
    return {ep: {p: float(np.median([r[p] for r in rows])) for p in P}
            for ep, rows in per.items()}


def render(res, ns, title, path, note=None):
    labs = list(res)
    nm = len(labs)
    with paper_style():
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.6), constrained_layout=True)
        x = np.arange(len(P))
        w = 0.8 / nm
        for j, lab in enumerate(labs):
            # colour only, matching the pushT contact figure: with a grey ramp the
            # lightness ordering already survives greyscale, so hatch adds no information.
            ax.bar(x + (j - (nm - 1) / 2) * w, [res[lab][p] for p in P], w,
                   color=series_color(j, nm), edgecolor="black", linewidth=0.5,
                   zorder=3, label=f"{lab} (n={ns[lab]})")
        ax.axhline(1.0, ls="--", color="0.45", lw=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(["above", "descend", "grasp", "lift", "carry", "place"])
        ax.set_ylabel(r"speedup $\alpha$ = demo / method")
        ax.set_title(title)
        top = max(v for m in res.values() for v in m.values() if np.isfinite(v))
        ax.set_ylim(0, top * 1.35)
        sparse_ticks(ax, "y", 4)
        ax.legend(ncol=3, loc="upper center", columnspacing=1.0, handletextpad=0.5)
        if note:
            ax.text(0.995, -0.30, note, transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color="0.35")
    save_fig(fig, path)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ep_alpha = {}
    for lab, runs in METHODS:
        live = [r for r in runs if (r / "rollout_traj.npz").is_file()]
        if not live:
            print(f"[{lab}] no runs on disk — skipped"); continue
        ep_alpha[lab] = per_phase_alpha(live)
        print(f"[{lab}] scored episodes: {len(ep_alpha[lab])}")

    labs = list(ep_alpha)
    # (a) every episode a method scored -- but each method is then measured on a
    #     DIFFERENT subset, and a method that failed the hard episodes looks faster.
    res_all = {l: {p: float(np.median([d[p] for d in ep_alpha[l].values()])) for p in P}
               for l in labs}
    ns_all = {l: len(ep_alpha[l]) for l in labs}
    # (b) only episodes every method solved -- paired, so the comparison is like-for-like.
    common = set.intersection(*(set(ep_alpha[l]) for l in labs))
    res_int = {l: {p: float(np.median([ep_alpha[l][e][p] for e in common])) for p in P}
               for l in labs}
    ns_int = {l: len(common) for l in labs}

    print(f"\nunion of scored episodes per method: " +
          "  ".join(f"{l}={ns_all[l]}" for l in labs))
    print(f"common to all {len(labs)} methods: {len(common)}\n")
    for name, res, ns in (("all", res_all, ns_all), ("intersection", res_int, ns_int)):
        print(f"-- {name}")
        for l in labs:
            print(f"   {l:<16} n={ns[l]:<4} " + "  ".join(f"{p}={res[l][p]:.2f}" for p in P))

    render(res_all, ns_all, "Cube: where the acceleration comes from",
           OUT / "phase_cube_table1.png",
           note="each method scored on every episode it solved — subsets differ")
    render(res_int, ns_int,
           f"Cube: per-phase speedup on the {len(common)} episodes every method solved",
           OUT / "phase_cube_table1_paired.png",
           note="paired: identical episode set for every method")


if __name__ == "__main__":
    main()
