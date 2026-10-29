#!/usr/bin/env python3
"""Paper figure: EPCC throughput (TP) against play-data size, one panel per task (1 x 3).

Figure layout: TP only, one row of three panels,
no legend, no suptitle, TP as the one shared y quantity, task names as panel titles.

Sized 1:1 like Fig. 3a (findings/260907_final_dtg_s1_warm/scripts/
plot_friction_plain3way_1to1.py, rendered as friction_plain3way_1to1_named.png):
3.54 x 1.8 in (a 0.495-textwidth subfigure of IEEEtran), every font at the caption's
8 pt, 400 dpi, no tight bbox, and the same line/marker weights and zero line.

Run on a CPU node (matplotlib/numpy are not allowed on the login node):
    sbatch --export=ALL,SCRIPT=findings/261001_scaling/scripts/plot_scaling_tp.py,\
ARGS="--denoms 256 128 64 32 16 4 1" scripts/analysis/run_analysis.sbatch

The Table I WM re-run (anchor) is off by default (not shown in the paper);
--anchor 1 draws it as a dashed dark-grey line, which the caption would then have to name.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_scaling import load, kfmt, FDIR  # noqa: E402  (also puts paper_style on sys.path)

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from paper_style import (paper_style, series_color, series_lw,  # noqa: E402
                         series_zorder, series_fill, sparse_ticks)

FIG_W = float(os.environ.get("FIG_W", 3.54))
FIG_H = float(os.environ.get("FIG_H", 1.3))     # 1.8 (Fig. 3a's height) made each panel ~1:1.8 tall
BOX = float(os.environ.get("BOX_ASPECT", 1.0))  # panel height / width: square panels
PT = float(os.environ.get("FIG_PT", 8))       # = IEEEtran caption size
RC_1TO1 = {"font.size": PT, "axes.titlesize": PT, "axes.labelsize": PT,
           "xtick.labelsize": PT, "ytick.labelsize": PT, "legend.fontsize": PT,
           "figure.dpi": 400, "savefig.dpi": 400}
TASKS = [("plain", "PushT"), ("slippery", "Slippery PushT"), ("cube", "Cube")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoms", type=int, nargs="+", default=[256, 128, 64, 32, 16, 4, 1])
    ap.add_argument("--anchor", type=int, default=0,   # off since 2026-10-05: user dropped it
                    help="1 = dashed line at the Table I re-run")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    data = {t: load(t, a.denoms) for t, _ in TASKS}
    frames = [p["frames"] for t, _ in TASKS for p in data[t][0]]
    xlim = (min(frames) / 1.6, max(frames) * 1.6)   # one x range for all three panels
    out = FDIR / "figures"
    out.mkdir(exist_ok=True)
    stem = f"scaling_tp_1x3{a.tag}"

    with paper_style(), mpl.rc_context(RC_1TO1):
        fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), constrained_layout=True)
        c = series_color(0, 2)
        for ax, (t, title) in zip(axes, TASKS):
            pts, anc = data[t]
            ax.plot([p["frames"] for p in pts], [p["TP"] for p in pts], color=c, marker="o",
                    ms=2.6, lw=series_lw(0) * 0.8, zorder=series_zorder(0),
                    mfc=series_fill(0), mec=c, mew=1.1)
            if a.anchor and anc is not None:
                ax.axhline(anc["TP"], color=series_color(1, 2), lw=series_lw(1) * 0.8,
                           ls="--", zorder=series_zorder(1))
            ax.set_box_aspect(BOX)
            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            # label 10k and 1M only: at 3.54 in a "100k" label runs into both neighbours,
            # and 100k stays readable as a minor tick between the two labelled decades
            ax.xaxis.set_major_locator(mticker.FixedLocator([1e4, 1e6]))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
            ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs="auto", numticks=12))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.set_title(title, fontsize=PT, pad=2.5)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.10 * (hi - lo))
            lo, hi = ax.get_ylim()
            if lo < 0.0 < hi:   # zero line only where TP crosses zero (not on Cube)
                ax.axhline(0, color="0.6", lw=0.8, ls=(0, (3, 3)), zorder=1)
            sparse_ticks(ax, "y", 3)
            ax.tick_params(pad=1.5)
        axes[0].set_ylabel("TP (1/s)", labelpad=1.5)
        axes[1].set_xlabel("play frames")   # one label under the middle panel, as in Fig. 3a
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{stem}.{ext}", dpi=400)   # no tight bbox: exact 1:1 canvas
    json.dump({"denoms": a.denoms, "anchor": bool(a.anchor), "size_in": [FIG_W, FIG_H], "pt": PT,
               "tasks": {t: {"points": data[t][0], "anchor": data[t][1]} for t, _ in TASKS}},
              open(out / f"{stem}_values.json", "w"), indent=1)
    print(f"wrote {out / f'{stem}.png'} (+pdf, values json)")


if __name__ == "__main__":
    main()
