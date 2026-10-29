# FINAL-TREE VARIANT (2026-09-08) of the snapshot plot_friction_plain3way.py: only paths/run maps changed.
# EPCC run dirs come from env EPCC_PUSHT / EPCC_CUBE / EPCC_SLIP (absolute paths); defaults = published rows.
"""Friction sweep with the baselines: ours vs DP / SuP / SAIL across damping.

Every row is on the same 50 initial conditions (seed 42,
`expert_slow_050_n200_d035`, walls-off), so the comparison is paired.

🔴 SAIL is scored at a TIME-MATCHED hold. `hold-50` counts 50 control STEPS,
and SAIL runs at control_ts ~0.057 s against everyone else's 0.100 s, so a flat
hold_k=50 would ask SAIL for a 2.8 s hold and everyone else for 5.0 s. Each
run's hold_k is therefore derived from its own control_ts as round(5.0 / ct),
which is 50 for every method except SAIL. The per-row hold_k is printed so the
correction is visible rather than buried.

Scored on the task criterion only. The `latched` panel (ever delivered to the
tolerance, regardless of holding) is deliberately not shown here — it separates
"reached the goal and could not settle" from "stopped delivering", which is a
diagnostic question rather than a performance one. It lives in
`friction_latched_table.py` for anyone who needs that split.

Styled to the group plot spec (v3): ours is series 0 — the only saturated colour,
heaviest line, filled markers, drawn on top — and the baselines run a dark-to-light grey
ramp with hollow markers behind it. Linestyle is NOT cycled: the ramp plus marker fill
already separate the series, and dashed greys read as noise at this panel size.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/baseline"))
import park_metrics as pm  # noqa: E402

# ---------------------------------------------------------------- 1:1 sizing --
# Authored at the size it is PLACED at (subfigure 0.495\textwidth of IEEEtran's
# 7.16 in text width = 3.54 in), with every font at the caption's 8 pt, so
# \includegraphics[width=\linewidth] applies scale 1.0 and the in-figure text
# matches the caption exactly. The _final variant is authored at DOUBLE_COL=6.9 in
# and is then squeezed to 0.51x in the PDF, which is why its labels read ~4.6 pt.
import os as _os, matplotlib as _mpl
FIG_W = float(_os.environ.get("FIG_W", 3.54))
FIG_H = float(_os.environ.get("FIG_H", 2.55))
PT    = float(_os.environ.get("FIG_PT", 8))       # = IEEEtran caption size
_RC1TO1 = {"font.size": PT, "axes.titlesize": PT, "axes.labelsize": PT,
           "xtick.labelsize": PT, "ytick.labelsize": PT, "legend.fontsize": PT,
           "figure.dpi": 400, "savefig.dpi": 400}
from paper_style import (paper_style, save_fig, series_color, series_lw,  # noqa: E402
                         series_zorder, series_fill, sparse_ticks, DOUBLE_COL)

BASE = Path(__file__).resolve().parents[2] / "findings/260521_baseline_setup"
FR = BASE / "logs_friction"
REPO = BASE.parents[1]
FINAL = REPO/"findings/260907_final_dtg_s1_warm"
OUT = FINAL / "fig_friction"
FR_NEW = FINAL / "runs/fig_friction"
DS = ["d000", "d010", "d020", "d030", "d040", "d050", "d060", "d070", "d080"]
D = np.array([int(d[1:]) / 100 for d in DS])
# The x axis is the block's velocity time constant, not the raw retention
# fraction. The engine multiplies velocity by d**dt per substep, which is the
# exact solution of vdot = -lambda v with lambda = ln(1/d), so tau = 1/lambda is
# the time the block keeps coasting after contact ends and v0*tau is the distance
# it covers doing so. d itself reads backwards (larger d = LESS friction) and has
# no units; tau reads forward and puts standard PushT at a finite point, tau = 0.
X = np.array([0.0 if d == 0 else 1.0 / np.log(1.0 / d) for d in D])
TARGET_S = 5.0
MIN_OK = 5

# (legend label, run prefix, short column key). ORDER IS THE STYLE: index 0 is ours and
# the module derives colour, weight, z-order and marker fill from that index -- so no
# per-series colour is hard-coded here. The k in the SAIL prefix stays (it names the run
# on disk) but is out of the label: only one k was ever swept for friction, so printing
# it invites the reader to look for a comparison that does not exist.
# Labels are the method names only. What each row is built from -- the shared plain-demo
# prior, ours additionally carrying a per-damping world model -- is a caption statement,
# not a legend entry; spelling it in the legend made the box wider than the panels.
SERIES = [   # (label, base dir, run prefix, key) — ours = the s1H84 + warm-ring redo, baselines untouched
    ("EPCC", FR_NEW, "fricB_s1warm_",   "oursB"),
    ("SuP",  FR,     "supplain_",       "SuPplain"),
    ("SAIL", FR,     "sailplain_k2.0_", "SAILplain"),
]


def score(run: Path):
    """hold-50-equivalent (time-matched), latched, TTC, TPR, and the hold_k used."""
    if not (run / "results.json").is_file():
        return None
    m1 = pm.park_metrics(run, hold_k=1)
    if m1 is None:
        return None
    ct = float(m1["results"].get("control_ts_mean", pm.DT))
    hk = max(1, int(round(TARGET_S / ct)))
    mh = pm.park_metrics(run, hold_k=hk)
    if mh is None:
        return None
    ok = mh["success"]
    t = mh["ttc_s"][ok]
    T = np.load(run / "rollout_traj.npz")["action"].shape[0]
    tmax = T * ct
    tpr = (sum(1.0 / max(x, 1e-9) for x in t)
           - (mh["n"] - int(ok.sum())) / tmax) / mh["n"]
    return dict(hold=float(ok.mean()), latch=float(m1["success"].mean()),
                ttc=float(np.median(t)) if t.size >= MIN_OK else np.nan,
                tpr=tpr, hk=hk, ct=ct)


def main():
    data = {lab: [score(base / f"{p}{d}") for d in DS] for lab, base, p, _ in SERIES}
    import json as _json
    OUT.mkdir(parents=True, exist_ok=True); (FINAL / "scores").mkdir(exist_ok=True)
    (FINAL / "scores/friction_scores.json").write_text(_json.dumps({lab: dict(zip(DS, v)) for lab, v in data.items()}, indent=1, default=str))
    g = lambda lab, k: np.array(  # noqa: E731
        [r[k] if r else np.nan for r in data[lab]], float)
    n = len(SERIES)

    with paper_style(), _mpl.rc_context(_RC1TO1):
        fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), constrained_layout=True)
        for i, (lab, _, _, _) in enumerate(SERIES):
            c = series_color(i, n)
            kw = dict(color=c, marker="o", ms=2.6, lw=series_lw(i) * 0.8,
                      zorder=series_zorder(i), mfc=series_fill(i), mec=c, mew=1.1)
            axes[0].plot(X, 100.0 * g(lab, "hold"), label=lab, **kw)   # SR in percent
            axes[1].plot(X, g(lab, "ttc"), **kw)
            axes[2].plot(X, g(lab, "tpr"), **kw)
        # Axis labels carry the quantity and its unit and nothing else. The scoring
        # qualifiers they used to spell out -- the 5 s hold, the median, SAIL Eq. 2 --
        # are definitions, not axis content; they live in this module's docstring and
        # in the printed table, and belong in the figure caption if the figure travels.
        for k, (ax, yl) in enumerate(zip(axes, ("SR (%)", "TTC (s)", "TP (1/s)"))):
            if _os.environ.get("TITLES", "0") == "1":
                ax.set_title(yl, fontsize=PT, pad=2.5)
            else:
                ax.set_ylabel(yl, labelpad=1.5)
            # one x range for all three: TTC exists only where a method still succeeds,
            # so autoscaling gave the middle panel a shorter axis and the same damping
            # sat at different screen positions in different panels.
            # The view stops at 3 s rather than at the last swept point (4.48 s). tau is
            # a nonlinear reparameterisation of a uniform d grid, so the top of the sweep
            # is sparse AND dead: every method is already at zero success by tau = 1.96,
            # and the two points beyond it repeat that with no new information while
            # eating half the panel. They are still plotted -- the curves run off the
            # right edge -- and the caption says where the sweep ends.
            ax.set_xlim(-0.15, 3.0)
            sparse_ticks(ax, "y", 3)
            sparse_ticks(ax, "x", 3)
            ax.tick_params(pad=1.5)
        axes[1].set_xlabel("coasting time (s)")   # one label under the middle panel:
        axes[0].set_ylim(0, 100)                  # three copies do not fit at 3.54 in,
                                                  # and supxlabel collides with the
                                                  # outside legend under constrained_layout
        lo, hi = axes[2].get_ylim()
        axes[2].set_ylim(lo - 0.10 * (hi - lo), hi + 0.10 * (hi - lo))
        axes[2].axhline(0, color="0.6", lw=0.8, ls=(0, (3, 3)), zorder=1)
        h, l = axes[0].get_legend_handles_labels()
        if _os.environ.get("NOLEG", "0") != "1":
            fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=PT,
                       fancybox=True, edgecolor="0.5", framealpha=0.9,
                       handletextpad=0.3, columnspacing=0.7, borderpad=0.25,
                       handlelength=1.3)
        # no suptitle: what the sweep is and what is held fixed belongs in the caption,
        # and at this width the title crowded the three panels
        fig.savefig(OUT / f"friction_plain3way_1to1{_os.environ.get('SUF','')}.png", dpi=400)  # no tight bbox: keep the exact
        # 3.54 x 2.55 in canvas so both subfigures land at the same height side by side

    print(f"{TARGET_S:.0f} s hold criterion · n=50 · same initial states\n")
    print(f"{'tau':>5s}" + "".join(f"{k:>11s}{'':>8s}{'':>8s}" for *_, k in SERIES))
    print(f"{'':>5s}" + "".join(f"{'SR':>11s}{'TTC':>8s}{'TPR':>8s}" for _ in SERIES))
    for i, d in enumerate(DS):
        row = ""
        for lab, *_ in SERIES:
            r = data[lab][i]
            row += (f"{r['hold']:>11.2f}{r['ttc']:>8.1f}{r['tpr']:>8.3f}"
                    if r else f"{'--':>11s}{'--':>8s}{'--':>8s}")
        print(f"{X[i]:>5.2f}" + row)
    print("\nhold_k applied (derived from control_ts):")
    for lab, *_ in SERIES:
        hs = sorted({r["hk"] for r in data[lab] if r})
        cs = sorted({round(r["ct"], 4) for r in data[lab] if r})
        print(f"  {lab:34s} ct={cs}  hold_k={hs}")
    print(f"\nsaved -> {OUT/'friction_plain3way.png'}")


if __name__ == "__main__":
    main()
