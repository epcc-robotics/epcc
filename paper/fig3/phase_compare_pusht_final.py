# FINAL-TREE VARIANT (2026-09-08) of the snapshot phase_compare_pusht.py: only paths/run maps changed.
# EPCC run dirs come from env EPCC_PUSHT / EPCC_CUBE / EPCC_SLIP (absolute paths); defaults = published rows.
#!/usr/bin/env python3
"""PushT / Slippery-PushT per-phase speedup, on the Table I baselines.

Phases (pushT has no oracle labels, so these are geometric):
    Approach  start -> tc,  tc = first step the block has moved > 3 px
    Push      tc -> ts,     ts = first step after tc with e < 2,
                            e(t) = max(pos_err / 20 px, ang_err / (pi/9))
    Settle    ts -> end

🔴 Three things the previous version (pusht_phase_compare_multi.py) got wrong, all of
which changed the answer:

1. ENDPOINT. It cut the rollout at `first_success_step` -- the LATCHED criterion, one
   brush of the goal -- while the paper scores hold-50. The gap is method-dependent:
   measured median hold50-minus-latched is +49 steps for ours on both tasks (the
   hold_k offset itself, i.e. arrive and stay) but +143 for SpeedupPatch on slippery.
   That is ~94 steps of drift-and-return the old metric never charged, all of it
   inside Settle. Now both the rollout and the demo end at their hold-50 completion.

2. SIM TIME. It compared raw step counts. SAIL varies its control period per step, so
   steps are not comparable across methods; phase duration now comes from the
   cumulative real period (park_metrics._time_axis, the same source the table uses).

3. PAIRING. Ratios were pooled over whatever each method solved, so a method that
   failed the hard episodes was measured on an easier set. Reported both ways.

4. DEMO REFERENCE. The alpha denominator needs a demonstrator that also satisfies
   hold-50, and the raw recordings often do not: only 57% (plain) / 80% (slippery) of
   them hold the goal for 50 steps before the recording ends. That is a recording-length
   artefact, not the demonstrator failing. The park-spliced twins hold 100% of the time
   and were built for exactly this; verified as a drop-in -- same 200 episodes, and
   block_pose/goal_pose identical over the original span (200/200), with a 100-frame
   oracle tail appended. Using the raw set instead silently drops ~40% of plain episodes.

Episodes are keyed by (episode_idx, start_step): the SAIL repeats use different seeds
and draw different inits, so episode_idx alone is not a unique problem id.
"""
import sys
from pathlib import Path

import os
import h5py
import hdf5plugin  # noqa
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "findings/260521_baseline_setup"
CACHE = Path(os.environ["STABLEWM_HOME"])
sys.path.insert(0, str(REPO / "scripts/baseline"))
import park_metrics as pm  # noqa
from paper_style import (paper_style, save_fig, series_color, series_hatch,  # noqa
                         sparse_ticks, DOUBLE_COL)

POS_T, ANG_T, MOVE, K = 20.0, np.pi / 9, 3.0, 2.0
HOLD, DEMO_CT = 50, pm.DT
P = ["approach", "push", "settle"]
PLABEL = ["Approach\n(free space)", "Push\n(transport)", "Settle\n(refine)"]
OUT = REPO/"findings/260907_final_dtg_s1_warm" / "fig_phase"
E = lambda k, d: Path(os.environ.get(k, str(d)))
SAILD = BASE / "logs_sail_reached"

TASKS = {
    "plain": dict(
        demo=CACHE / "datasets/expert_slow_025_n200_centered_v2_park.h5",
        methods=[("EPCC (ours)",  [E("EPCC_PUSHT", BASE / "logs_fixstats/fix_plainM2_current")]),
                 ("SpeedupPatch", [BASE / "logs_sh_repro_wallsoff/pusht_sup_paperhp"]),
                 ("SAIL $k{=}2$", [SAILD / f"plain_sail_awe_per_env_k2.0_n50_r{r}" for r in (1, 2, 3)]),
                 ("SAIL $k{=}3$", [SAILD / f"plain_sail_awe_per_env_k3.0_n50_r{r}" for r in (1, 2, 3)]),
                 ("SkiP",         [BASE / "logs_sh_repro_wallsoff/pusht_skip_wallsoff"])]),
    "slippery": dict(
        demo=CACHE / "slippery_pusht/data/expert_slow_050_n200_d035_park.h5",
        methods=[("EPCC (ours)",  [E("EPCC_SLIP", BASE / "logs_epcc5/rfx_slipM1_nu15asm1")]),
                 ("SpeedupPatch", [BASE / "logs_sh_repro_wallsoff/slippery_sup_paperhp_rel"]),
                 ("SAIL $k{=}2$", [SAILD / f"slippery_sail_awe_per_env_k2.0_n50_r{r}" for r in (1, 2, 3)]),
                 ("SAIL $k{=}3$", [SAILD / f"slippery_sail_awe_per_env_k3.0_n50_r{r}" for r in (1, 2, 3)]),
                 ("SkiP",         [BASE / "logs_sh_repro_wallsoff/slippery_pusht_skip_wallsoff"])]),
}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def bounds(path, goal, k=K):
    """(tc, ts, end) step indices. end is the last index of the window."""
    n = len(path)
    disp = np.linalg.norm(path[:, :2] - path[0, :2], axis=1)
    tc = int(np.argmax(disp > MOVE)) if (disp > MOVE).any() else 0
    e = np.maximum(np.linalg.norm(path[:, :2] - goal[:2], axis=1) / POS_T,
                   np.abs(wrap(path[:, 2] - goal[2])) / ANG_T)
    below = np.where(e[tc:] < k)[0]
    ts = (tc + int(below[0])) if len(below) else n - 1
    return tc, ts, n - 1


def per_episode_alpha(runs, demo_h5):
    """(episode_idx, start_step) -> {phase: alpha}. Both clocks are hold-50."""
    with h5py.File(demo_h5, "r") as f:
        ep_off, ep_len = f["ep_offset"][:].astype(int), f["ep_len"][:].astype(int)
        dbp = f["block_pose"][:].astype(np.float64)
        dgp = f["goal_pose"][:].astype(np.float64)
    dframe = pm.demo_park_time(str(demo_h5), hold_k=HOLD, mode="first", ttc_at="complete")

    acc = {}
    for run in runs:
        m = pm.park_metrics(run, hold_k=HOLD, mode="first", ttc_at="complete")
        if m is None:
            print(f"    [skip] {run.name}: park_metrics returned None"); continue
        z = np.load(run / "rollout_traj.npz")
        bpr, gstate = z["block_pose"], z["goal_state"]
        epi, s0 = m["episode_idx"], m["start_step"]
        T, n = bpr.shape[0], bpr.shape[1]
        cum = pm._time_axis(run, T, n, float(m["results"].get("control_ts_mean", DEMO_CT)),
                            m["results"])
        if cum is None:
            print(f"    [skip] {run.name}: no honest time axis"); continue
        for i in range(n):
            if not m["success"][i]:
                continue
            ep, st = int(epi[i]), int(s0[i])
            df = dframe[ep]
            if df < 0 or df - st < 4:
                continue
            end = int(min(m["park_start"][i] + HOLD, T - 1))
            if end < 6:
                continue
            # demo side: from the eval's reset frame to the demo's own hold-50 park
            o = ep_off[ep]
            dpath = dbp[o + st: o + int(df) + 1]
            if len(dpath) < 4:
                continue
            db = bounds(dpath, dgp[o])
            ob = bounds(bpr[:end + 1, i, :], gstate[2:5])
            dseg = [db[0], db[1] - db[0], db[2] - db[1]]
            if any(s <= 0 for s in dseg):
                continue
            lo_hi = [(0, ob[0]), (ob[0], ob[1]), (ob[1], ob[2])]
            if any(b - a <= 0 for a, b in lo_hi):
                continue
            row = {}
            for j, p in enumerate(P):
                a, b = lo_hi[j]
                o_s = float(cum[b, i] - cum[a, i])
                if o_s <= 0:
                    row = None; break
                row[p] = (dseg[j] * DEMO_CT) / o_s
            if row is not None:
                acc.setdefault((ep, st), []).append(row)
    return {k2: {p: float(np.median([r[p] for r in rows])) for p in P}
            for k2, rows in acc.items()}


def render(res, ns, title, path, note):
    labs = list(res)
    nm = len(labs)
    with paper_style():
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.5), constrained_layout=True)
        x = np.arange(len(P)); w = 0.8 / nm
        for j, lab in enumerate(labs):
            ax.bar(x + (j - (nm - 1) / 2) * w, [res[lab][p] for p in P], w,
                   color=series_color(j, nm), hatch=series_hatch(j),
                   edgecolor="black", linewidth=0.5, zorder=3,
                   label=f"{lab} (n={ns[lab]})")
        ax.axhline(1.0, ls="--", color="0.45", lw=0.8, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels(PLABEL)
        ax.set_ylabel(r"speedup $\alpha$ = demo / method")
        ax.set_title(title)
        top = max(v for mm in res.values() for v in mm.values() if np.isfinite(v))
        ax.set_ylim(0, top * 1.35)
        sparse_ticks(ax, "y", 4)
        ax.legend(ncol=3, loc="upper center", columnspacing=1.0, handletextpad=0.5)
        ax.text(0.995, -0.34, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color="0.35")
    save_fig(fig, path); plt.close(fig)
    print(f"[plot] wrote {path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for task, cfg in TASKS.items():
        print(f"########## {task}")
        ep_a = {}
        for lab, runs in cfg["methods"]:
            live = [r for r in runs if (r / "rollout_traj.npz").is_file()]
            if not live:
                print(f"  {lab}: no runs"); continue
            ep_a[lab] = per_episode_alpha(live, cfg["demo"])
            print(f"  {lab:<15} scored {len(ep_a[lab])} episodes")
        labs = [l for l, _ in cfg["methods"] if l in ep_a and ep_a[l]]
        if not labs:
            continue
        res_all = {l: {p: float(np.median([d[p] for d in ep_a[l].values()])) for p in P}
                   for l in labs}
        ns_all = {l: len(ep_a[l]) for l in labs}
        common = set.intersection(*(set(ep_a[l]) for l in labs))
        print(f"  common to all {len(labs)} methods: {len(common)}")
        # One method with a low success rate can starve the shared set for everyone --
        # on slippery, SkiP solves 8 episodes and drags the 5-way intersection to 4.
        # Drop the scarcest method rather than abandon the paired view, and say so.
        pair_labs, dropped = list(labs), []
        while len(common) < 8 and len(pair_labs) > 2:
            worst = min(pair_labs, key=lambda l: len(ep_a[l]))
            dropped.append(f"{worst} (n={len(ep_a[worst])})")
            pair_labs.remove(worst)
            common = set.intersection(*(set(ep_a[l]) for l in pair_labs))
            print(f"    dropped {worst} -> intersection {len(common)}")
        for name, res, ns in (("all", res_all, ns_all),):
            for l in labs:
                print(f"    {l:<15} n={ns[l]:<4} " + "  ".join(f"{p}={res[l][p]:.2f}" for p in P))
        render(res_all, ns_all, f"PushT {task}: where the acceleration comes from",
               OUT / f"phase_{task}_table1.png",
               "each method scored on every episode it solved — subsets differ")
        if len(common) >= 8:
            res_int = {l: {p: float(np.median([ep_a[l][e][p] for e in common])) for p in P}
                       for l in pair_labs}
            print("    -- intersection")
            for l in pair_labs:
                print(f"    {l:<15} n={len(common):<4} " +
                      "  ".join(f"{p}={res_int[l][p]:.2f}" for p in P))
            note = "paired: identical episode set for every method"
            if dropped:
                note += "  |  excluded for scarcity: " + ", ".join(dropped)
            render(res_int, {l: len(common) for l in pair_labs},
                   f"PushT {task}: per-phase speedup on the {len(common)} shared episodes",
                   OUT / f"phase_{task}_table1_paired.png", note)
        else:
            print(f"    intersection too small ({len(common)}) — paired figure skipped")
        print()


if __name__ == "__main__":
    main()
