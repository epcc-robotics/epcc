#!/usr/bin/env python3
"""Data-scaling figure: EPCC SR / TTC / TP vs the amount of play data the WM learns from.

Run on a CPU node (matplotlib/numpy are not allowed on the login node):
    sbatch --export=ALL,SCRIPT=findings/261001_scaling/scripts/plot_scaling.py,ARGS="--task plain" \
           scripts/analysis/run_analysis.sbatch
    ARGS="--task all --denoms 256 128 64 32 16 4 1 --tag _7pt"   # 3x3 grid: rows = tasks

Inputs
  $STABLEWM_HOME/datasets/scaling/<task>/<prefix>_manifest.json   actual subset frame counts
  findings/261001_scaling/scores/pilot_<task>__ev_<task>_f<D>.json  scored EPCC evals (score_runs.py)
  findings/261001_scaling/scores/anchor__anchor_<task>_tableI.json  Table I cell re-run, same session
Output
  findings/261001_scaling/figures/scaling_<task><tag>.{png,pdf} + scaling_<task><tag>_values.json
  (<task> = all for the 3x3 grid)

Style: scripts/baseline/paper_style.py (v3). Ours = the sweep (index 0, blue, filled);
the Table I WM re-run is a grey dashed reference line. TTC is drawn only for points with
at least 5 successes (the Fig. 3a convention); fewer successes make the median meaningless.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/baseline"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from paper_style import (paper_style, save_fig, series_color, series_lw,  # noqa: E402
                         series_zorder, series_fill, sparse_ticks, DOUBLE_COL)

PREFIX = {"plain": "plain_main2", "slippery": "slippery_main2", "cube": "cube_main2"}
TITLE = {"plain": "PushT", "slippery": "Slippery-PushT", "cube": "Cube"}
PANELS = [("SR", "SR (%)"), ("TTC", "TTC (s)"), ("TP", "TP (1/s)")]
MIN_SUCCESS_FOR_TTC = 5
HOME = Path(os.environ["STABLEWM_HOME"])
FDIR = REPO / "findings/261001_scaling"


def kfmt(x, _pos=None):
    if x >= 1e6:
        return f"{x / 1e6:g}M"
    if x >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def load(task, denoms):
    """Scored sweep points of one task (sorted by frames) and its Table I anchor."""
    man = json.load(open(HOME / f"datasets/scaling/{task}/{PREFIX[task]}_manifest.json"))
    pts = []
    for d in denoms:
        sp = FDIR / f"scores/pilot_{task}__ev_{task}_f{d}.json"
        if not sp.exists():
            print(f"[skip] {task} 1/{d}: no score file yet ({sp.name})")
            continue
        r = json.load(open(sp))["rows"][0]
        pts.append({"denom": d, "frames": man["subsets"][str(d)]["frames"],
                    "SR": r["SR_pct"], "TTC": r["TTC_median_s"], "TP": r["TP"],
                    "n_success": r["n_success"], "n": r["n"], "run": r["run"]})
    pts.sort(key=lambda p: p["frames"])
    anc = None
    ap_ = FDIR / f"scores/anchor__anchor_{task}_tableI.json"
    if ap_.exists():
        r = json.load(open(ap_))["rows"][0]
        anc = {"SR": r["SR_pct"], "TTC": r["TTC_median_s"], "TP": r["TP"], "wm": r["meta"]["wm"]}
    print(f"== {TITLE[task]}")
    for p in pts:
        print(f"1/{p['denom']:<4} frames {p['frames']:>9,}  SR {p['SR']:5.1f}  TTC {p['TTC']}  "
              f"TP {p['TP']:+.4f}  ({p['n_success']}/{p['n']})")
    if anc:
        print(f"anchor (Table I {anc['wm']} re-run): SR {anc['SR']}  TTC {anc['TTC']}  TP {anc['TP']:+.4f}")
    return pts, anc


def draw_row(axes, pts, anc, titles=True, xlabel=True, xlim=None):
    """One task: SR / TTC / TP panels against play frames (log x)."""
    if xlim is None:
        xlim = (min(p["frames"] for p in pts) / 1.6, max(p["frames"] for p in pts) * 1.6)
    for ax, (key, title) in zip(axes, PANELS):
        use = [p for p in pts if key != "TTC" or p["n_success"] >= MIN_SUCCESS_FOR_TTC]
        xs = [p["frames"] for p in use]
        ys = [p[key] for p in use]
        ax.plot(xs, ys, marker="o", ms=4, color=series_color(0, 2), lw=series_lw(0),
                zorder=series_zorder(0), mfc=series_fill(0), mec=series_color(0, 2),
                mew=1.1, label="EPCC, WM on subset of D")
        if anc is not None:
            ax.axhline(anc[key], color=series_color(1, 2), lw=series_lw(1), ls="--",
                       zorder=series_zorder(1), label="Table I WM (re-run)")
        if key == "TP":
            ax.axhline(0.0, color="0.6", lw=0.6, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        if titles:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel("play frames")
        ax.margins(y=0.12)
        if key == "SR":
            ax.set_ylim(0, 105)
        elif key == "TTC" and ys:
            # autoscale would stretch a 0.1 s spread (cube) across the whole panel;
            # show at least -20% / +20% around the plotted values and the anchor
            vals = ys + ([anc["TTC"]] if anc is not None and anc["TTC"] is not None else [])
            ax.set_ylim(min(vals) * 0.8, max(vals) * 1.2)
        sparse_ticks(ax, "y", 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(PREFIX) + ["all"])
    ap.add_argument("--denoms", type=int, nargs="+", default=[256, 64, 16, 4, 1])
    ap.add_argument("--tag", default="", help="output-name suffix, e.g. _7pt")
    a = ap.parse_args()

    tasks = list(PREFIX) if a.task == "all" else [a.task]
    data = {t: load(t, a.denoms) for t in tasks}
    out = FDIR / "figures"
    out.mkdir(exist_ok=True)
    stem = f"scaling_{a.task}{a.tag}"
    with paper_style():
        if a.task == "all":
            fig, axes = plt.subplots(len(tasks), 3, figsize=(DOUBLE_COL, 1.85 * len(tasks)),
                                     constrained_layout=True)
            # one x range for every row, so the tasks' transitions compare in absolute frames
            frames = [p["frames"] for t in tasks for p in data[t][0]]
            xlim = (min(frames) / 1.6, max(frames) * 1.6)
            for i, t in enumerate(tasks):
                draw_row(axes[i], *data[t], titles=(i == 0), xlabel=(i == len(tasks) - 1),
                         xlim=xlim)
                axes[i][0].set_ylabel(TITLE[t], fontweight="bold")
            fig.suptitle("Data scaling: EPCC with the WM trained on nested subsets of D",
                         fontsize=10, fontweight="bold")
        else:
            fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.1), constrained_layout=True)
            draw_row(axes, *data[a.task])
            fig.suptitle(f"{TITLE[a.task]}: data scaling", fontsize=10, fontweight="bold")
            axes = [axes]
        # one shared legend below the panels: inside any panel it covers a data point
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels),
                   handlelength=1.8)
        for ext in ("png", "pdf"):
            save_fig(fig, out / f"{stem}.{ext}")
    json.dump({"task": a.task, "denoms": a.denoms, "ttc_min_success": MIN_SUCCESS_FOR_TTC,
               "tasks": {t: {"points": data[t][0], "anchor": data[t][1]} for t in tasks}},
              open(out / f"{stem}_values.json", "w"), indent=1)
    print(f"wrote {out / f'{stem}.png'} (+pdf, values json)")


if __name__ == "__main__":
    main()
