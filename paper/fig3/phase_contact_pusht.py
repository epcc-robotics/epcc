#!/usr/bin/env python3
"""PushT split by CONTACT, not by where the block ended up.

The geometric split (Approach / Push / Settle) put its Push->Settle boundary at the moment
the block entered twice the goal tolerance -- an outcome the agent cannot observe, and one
that produces no visible change in behaviour. This splits on what the agent is doing:

    CONTACT       the agent is touching the T
    REPOSITION    it is not -- travelling to the next push site

which is the same axis the cube split uses (grasp/lift are contact) and the same axis the
paper argues about (free space vs contact-rich).

The rollout dumps carry no contact flag, so contact is computed geometrically: the agent
is a disc of radius 15 (base_radius 0.375 x scale 40, env.py add_circle) and the T is two
axis-aligned boxes in the block frame (scale 30, length 4, env.py add_tee). The demo h5
DOES record the simulator's own `n_contacts`, so the geometric test is calibrated and
scored against it before being used anywhere.
"""
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/baseline"))
sys.path.insert(0, str(REPO / "Ablation/B_accelerate"))
import park_metrics as pm  # noqa
from paper_style import (paper_style, save_fig, series_color, series_hatch,  # noqa
                         sparse_ticks, DOUBLE_COL)
from phase_compare_pusht import TASKS, HOLD, DEMO_CT  # noqa

OUT = REPO / "findings/260521_baseline_setup/ablation_b"
GROUPS = ["contact", "reposition"]
GLABEL = ["CONTACT\n(pushing)", "REPOSITION\n(travelling)"]

R_AGENT = 0.375 * 40          # env.py: base_radius * agent scale
S, L = 30.0, 4                # env.py add_tee
RECTS = [(-L*S/2, L*S/2, 0.0, S), (-S/2, S/2, S, L*S)]   # (x0,x1,y0,y1) in block frame


def dist_to_tee(agent_xy, block_xy, ang):
    """Distance from agent centre to the T surface, vectorised over time."""
    c, s = np.cos(-ang), np.sin(-ang)
    d = agent_xy - block_xy
    lx = c * d[:, 0] - s * d[:, 1]
    ly = s * d[:, 0] + c * d[:, 1]
    best = None
    for x0, x1, y0, y1 in RECTS:
        dx = np.maximum(np.maximum(x0 - lx, 0.0), lx - x1)
        dy = np.maximum(np.maximum(y0 - ly, 0.0), ly - y1)
        r = np.hypot(dx, dy)
        best = r if best is None else np.minimum(best, r)
    return best


def calibrate(h5):
    """Pick the contact tolerance that best reproduces the simulator's own n_contacts."""
    with h5py.File(h5, "r") as f:
        off, ln = f["ep_offset"][:].astype(int), f["ep_len"][:].astype(int)
        bp, ap = f["block_pose"][:], f["pos_agent"][:]
        nc = f["n_contacts"][:]
    sel = slice(0, min(len(bp), 40000))
    d = dist_to_tee(ap[sel], bp[sel, :2], bp[sel, 2])
    truth = nc[sel] > 0
    # 🔴 The grid must be fine near zero. pymunk holds a resting contact at a gap of
    # ~0 px, and a strict `<` drops exactly those frames -- on slippery that was half of
    # all contact (6.0% predicted vs 12.3% recorded). A 0.5-px grid stepped straight over
    # the optimum at 0.05.
    best = None
    for tol in [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0]:
        pred = d <= R_AGENT + tol
        agree = float(np.mean(pred == truth))
        if best is None or agree > best[1]:
            best = (float(tol), agree, float(np.mean(pred)), float(np.mean(truth)))
    return best


def segments(mask):
    """How many separate contact bouts, ignoring 1-step flicker."""
    m = np.asarray(mask, bool)
    return int(np.sum(m[1:] & ~m[:-1])) + int(m[0])


RESULTS = {}
RESULTS_ALL = {}
RESULTS_STRICT = {}


def main():
    for task, cfg in TASKS.items():
        tol, agree, pf, tf = calibrate(cfg["demo"])
        print(f"########## {task}")
        print(f"  contact test calibrated on the simulator's n_contacts: "
              f"tol {tol:+.1f} px -> agreement {agree:.1%}  "
              f"(predicted contact {pf:.1%} of frames, recorded {tf:.1%})")
        THR = R_AGENT + tol

        with h5py.File(cfg["demo"], "r") as f:
            ep_off, ep_len = f["ep_offset"][:].astype(int), f["ep_len"][:].astype(int)
            dbp, dap = f["block_pose"][:], f["pos_agent"][:]
        dframe = pm.demo_park_time(str(cfg["demo"]), hold_k=HOLD, mode="first", ttc_at="complete")

        ep_a = {}
        for label, runs in cfg["methods"]:
            acc = {}
            for run in runs:
                if not (run / "rollout_traj.npz").is_file():
                    continue
                m = pm.park_metrics(run, hold_k=HOLD, mode="first", ttc_at="complete")
                if m is None:
                    continue
                z = np.load(run / "rollout_traj.npz")
                bpr, ag = z["block_pose"], z["agent_pos"]
                T, n = bpr.shape[0], bpr.shape[1]
                cum = pm._time_axis(run, T, n, float(m["results"].get("control_ts_mean", DEMO_CT)),
                                    m["results"])
                if cum is None:
                    continue
                for i in range(n):
                    if not m["success"][i]:
                        continue
                    ep, st = int(m["episode_idx"][i]), int(m["start_step"][i])
                    df = dframe[ep]
                    if df < 0 or df - st < 4:
                        continue
                    end = int(min(m["park_start"][i] + HOLD, T - 1))
                    if end < 6:
                        continue
                    o = ep_off[ep]
                    dsl = slice(o + st, o + int(df) + 1)
                    dcon = dist_to_tee(dap[dsl], dbp[dsl, :2], dbp[dsl, 2]) <= THR
                    ocon = dist_to_tee(ag[:end+1, i, :], bpr[:end+1, i, :2], bpr[:end+1, i, 2]) <= THR
                    dt = np.full(dcon.shape, DEMO_CT)
                    ostep = np.diff(cum[:end+2, i]) if cum.shape[0] > end+1 else np.diff(cum[:end+1, i])
                    ostep = ostep[:len(ocon)]
                    d_c, d_r = float(dt[dcon].sum()), float(dt[~dcon].sum())
                    o_c, o_r = float(ostep[ocon[:len(ostep)]].sum()), float(ostep[~ocon[:len(ostep)]].sum())
                    if min(d_c, d_r, o_c, o_r) <= 0:
                        continue
                    acc.setdefault((ep, st), []).append(
                        dict(contact=d_c/o_c, reposition=d_r/o_r,
                             cfrac=float(ocon.mean()), bouts=segments(ocon),
                             dcfrac=float(dcon.mean()), dbouts=segments(dcon)))
            if acc:
                ep_a[label] = {k: {kk: float(np.median([r[kk] for r in v])) for kk in v[0]}
                               for k, v in acc.items()}
        labs = list(ep_a)
        if not labs:
            print("  nothing scored"); continue
        common = set.intersection(*(set(ep_a[l]) for l in labs))
        print(f"  {'method':<15}{'n':>4}{'a_contact':>11}{'a_reposition':>14}"
              f"{'contact %':>11}{'bouts':>7}   (demo: contact % / bouts)")
        for l in labs:
            v = ep_a[l]
            g = lambda k: np.median([d[k] for d in v.values()])  # noqa
            print(f"  {l:<15}{len(v):>4}{g('contact'):>11.2f}{g('reposition'):>14.2f}"
                  f"{g('cfrac'):>11.0%}{g('bouts'):>7.0f}   ({g('dcfrac'):.0%} / {g('dbouts'):.0f})")
        # strict: every method kept, even one that starves the shared set (slippery SkiP
        # solves 12, which drops the 5-way intersection to 4 -- reported, not hidden)
        strict = set.intersection(*(set(ep_a[l]) for l in labs))
        if strict:
            RESULTS_STRICT[task] = ({l: {k: float(np.median([ep_a[l][e][k] for e in strict]))
                                         for k in GROUPS} for l in labs}, len(strict))
        # a scarce method starves the shared set for everyone; drop it and say so
        pair_labs, dropped = list(labs), []
        while len(common) < 8 and len(pair_labs) > 2:
            worst = min(pair_labs, key=lambda l: len(ep_a[l]))
            dropped.append(f"{worst} (n={len(ep_a[worst])})")
            pair_labs.remove(worst)
            common = set.intersection(*(set(ep_a[l]) for l in pair_labs))
        if len(common) >= 8:
            print(f"  -- paired on the {len(common)} shared episodes"
                  + (f"   [excluded: {', '.join(dropped)}]" if dropped else ""))
            res = {}
            for l in pair_labs:
                v = ep_a[l]
                res[l] = {k: float(np.median([v[e][k] for e in common])) for k in GROUPS}
                print(f"  {l:<15}{len(common):>4}{res[l]['contact']:>11.2f}"
                      f"{res[l]['reposition']:>14.2f}")
            RESULTS[task] = (res, len(common), dropped)
        # unpaired: every episode a method solved. Subsets differ, but nothing is dropped
        # for scarcity, so SkiP survives and slippery keeps its full sample.
        RESULTS_ALL[task] = ({l: {k: float(np.median([d[k] for d in ep_a[l].values()]))
                                  for k in GROUPS} for l in labs},
                             {l: len(ep_a[l]) for l in labs})
        print()

    def draw(data, path, note, ns_of):
        with paper_style():
            fig, axes = plt.subplots(1, len(data), figsize=(DOUBLE_COL, 3.0),
                                     constrained_layout=True, squeeze=False)
            # colour must key off the method, not its position in this panel: slippery
            # drops SkiP, and indexing by panel position would repaint every method.
            order = [l for l, _ in TASKS["plain"]["methods"]]
            NM = len(order)
            for ax, (task, res) in zip(axes[0], data.items()):
                labs = [l for l in order if l in res]
                nm = len(labs)
                x = np.arange(len(GROUPS)); w = 0.8 / nm
                for j, l in enumerate(labs):
                    k = order.index(l)
                    # colour only: with a grey ramp the lightness ordering already survives
                    # greyscale, so hatch adds texture without adding information.
                    ax.bar(x + (j - (nm - 1) / 2) * w, [res[l][g] for g in GROUPS], w,
                           color=series_color(k, NM), edgecolor="black",
                           linewidth=0.5, zorder=3, label=l)
                ax.axhline(1.0, ls="--", color="0.45", lw=0.8, zorder=2)
                ax.set_xticks(x); ax.set_xticklabels(GLABEL)
                ax.set_title(f"PushT {task}   ({ns_of(task, res)})", loc="left", fontsize=9.5)
                top = max(v for m2 in res.values() for v in m2.values())
                ax.set_ylim(0, top * 1.35); sparse_ticks(ax, "y", 4)
            axes[0][0].set_ylabel(r"speedup $\alpha$ = demo / method")
            h, lb = axes[0][0].get_legend_handles_labels()
            # "outside" keeps constrained_layout from letting the legend sit on the titles
            fig.legend(h, lb, ncol=5, loc="outside lower center",
                       columnspacing=1.2, handletextpad=0.5, fontsize=7.5)
        save_fig(fig, path)
        plt.close(fig)
        print(f"[plot] wrote {path}")

    if RESULTS:
        OUT.mkdir(parents=True, exist_ok=True)
        draw({t: r for t, (r, n, d) in RESULTS.items()},
             OUT / "phase_contact_pusht_paired.png", "paired",
             lambda t, r: (f"n = {RESULTS[t][1]}" +
                           (f", {', '.join(d.split(' (')[0] for d in RESULTS[t][2])} excluded"
                            if RESULTS[t][2] else "")))
    if RESULTS_ALL:
        draw({t: r for t, (r, ns) in RESULTS_ALL.items()},
             OUT / "phase_contact_pusht_all.png", "all",
             lambda t, r: "n = " + "/".join(str(RESULTS_ALL[t][1][l]) for l in r))
    if RESULTS_STRICT:
        draw({t: r for t, (r, n) in RESULTS_STRICT.items()},
             OUT / "phase_contact_pusht_paired_all5.png", "strict",
             lambda t, r: f"n = {RESULTS_STRICT[t][1]}, all 5 methods")
        for t, (r, n) in RESULTS_STRICT.items():
            print(f"  [strict {t}] n={n}  " +
                  "  ".join(f"{l}: {r[l]['contact']:.2f}/{r[l]['reposition']:.2f}" for l in r))


if __name__ == "__main__":
    main()
