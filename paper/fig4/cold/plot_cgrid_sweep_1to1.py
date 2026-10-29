# 2026-10-22 COLD-PREDICTOR VARIANT of findings/260907_final_dtg_s1_warm/scripts/plot_cgrid_sweep_1to1.py.
# Only the run map and output paths change: PushT/Slippery points are the f4cold reruns (predictor from
# scratch, 100 epochs, lowest validation prediction loss; Slippery CTRL scaler = its own training set),
# Cube DEMO/CTRL are the published runs (already cold), Cube D points are re-evaluated at their
# lowest-validation epoch. Outputs go to findings/261022_fig4_cold/{fig_expert_play,scores}.
# EPCC run dirs come from env EPCC_PUSHT / EPCC_CUBE / EPCC_SLIP (absolute paths); defaults = published rows.
# champion rows are the OLD published ones (measured & printed, never drawn).
# SHOW_PREV=1 (default) overlays the published cells faintly; tasks with no campaign runs are skipped.
"""Two figures for the collection-pace question.

(1) cgrid_marginals.png — the sanity check: where in the state space does each corpus
    actually live? 2D block-position histograms, one column per corpus, shared colour
    scale. Read from `state[:, 2:4]`; the `block_pose` field is ALL-ZERO for play
    episodes in these corpora, so using it silently compares zeros against real poses.

(2) cgrid_sweep.png — the bar chart re-drawn as a sweep. The paced corpora and the
    control corpus are the same thing at different collection paces, so they share an
    x axis: `make_mix_manifests.py` tags the play corpus c=1.0 and the D grid 0.5/1.5/2.5
    (cube 0.3/0.75/1.2). Demo-only is not play at any pace, so it is a horizontal
    reference line rather than a point on the sweep. Champion is measured and printed for
    provenance but NOT drawn -- it is the same c=1.0 corpus at 12-28x the frames, which
    is a data-volume comparison and belongs to a different question than this figure's.

🔴 The c=1.0 point is the CONTROL cell, collected in the original play campaign rather
than the D campaign. A `D_c1.0_final.h5` exists on disk but no world model was trained on
it, so the sweep cannot yet be closed within one campaign. The c=1.0 re-collection was
verified against the original play to <3% at gate time, which is why substituting it is
defensible -- but it is a substitution and is drawn hollow to say so.
"""
import sys
from pathlib import Path

import h5py, hdf5plugin  # noqa
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/baseline"))
sys.path.insert(1, str(REPO / "findings/260907_final_dtg_s1_warm/scripts"))   # paper_style
import park_metrics as pm  # noqa

# ---------------------------------------------------------------- 1:1 sizing --
# Authored at the size it is PLACED at, with every font at the caption's 8 pt, so
# \includegraphics[width=\linewidth] applies scale 1.0. Panel proportions follow the
# original 6.9 x 4.9 in layout (FIG_H defaults to FIG_W * 4.9/6.9).
import os as _os, matplotlib as _mpl
FIG_W = float(_os.environ.get("FIG_W", 7.16))
FIG_H = float(_os.environ.get("FIG_H", 7.16 * 4.9 / 6.9))
PT    = float(_os.environ.get("FIG_PT", 8))
_RC1TO1 = {"font.size": PT, "axes.titlesize": PT, "axes.labelsize": PT,
           "xtick.labelsize": PT, "ytick.labelsize": PT, "legend.fontsize": PT,
           "figure.dpi": 400, "savefig.dpi": 400}
from paper_style import (paper_style, save_fig, series_color, sparse_ticks,  # noqa
                         DOUBLE_COL)

B = REPO / "findings/260521_baseline_setup"
FX, AGG, WO = B / "logs_fixstats", B / "logs_agg_eval", B / "logs_sh_repro_wallsoff"
OLD = REPO / "findings/260907_final_dtg_s1_warm/runs/fig_expert_play"   # published cells
FINAL = REPO / "findings/261022_fig4_cold"
RUNS = FINAL / "runs/fig_expert_play"
OUT = FINAL / "fig_expert_play"
H = Path(_os.environ["STABLEWM_HOME"])
HOLD, NB = 50, 26
# Demo-only has no place on the pace axis -- it is a different corpus, not play collected
# faster or slower -- so it is a horizontal reference across the panel. Its own collection
# pace is still in each demo file's name (expert_slow_025 -> 0.25, expert_slow_050 -> 0.50,
# cube ..._c0.3_... -> 0.30) and is printed in the audit, just not encoded as an x position.
DEMO_C = {"Plain PushT": 0.25, "Slippery PushT": 0.50, "Cube": 0.30}
# Panel titles: the paper's task names, identical across Fig. 3a/3b/4.
DISP = {"Plain PushT": "PushT", "Slippery PushT": "Slippery-PushT", "Cube": "Cube"}

# Measured pace: the 90th percentile of each corpus's per-step command magnitude,
# divided by the demonstration set's own. Requesting a collection scale c is not the
# same as realising it -- the commands compress at large c and saturate outright on
# Cube -- so the axis reports what the data contains, not what was asked for.
# Source: scripts/pace_audit.py, logs_smoke/pace_audit3_13116783.out.
XMEAS = {
    "Plain PushT":     {0.5: 1.85, 1.0: 3.51, 1.5: 4.15, 2.5: 5.30},
    "Cube":            {0.3: 1.06, 0.75: 2.65, 1.0: 3.54, 1.2: 4.22},
    "Slippery PushT":  {0.5: 1.44, 1.0: 2.47, 1.5: 3.28, 2.5: 4.21},
}
MEASURED_X = _os.environ.get("MEASURED_X", "0") == "1"


def xval(task, c):
    """Collection scale -> plotted abscissa."""
    if not MEASURED_X:
        return c
    m = XMEAS.get(task, {})
    if c not in m:
        raise KeyError(f"no measured pace for {task} c={c}; rerun pace_audit.py")
    return m[c]

# (task, cube?) -> dict(champion=, control=(c, run), demo=, paced=[(c, run), ...])
CELLS = {
    "Plain PushT": dict(cube=False,
        champion=WO / "pusht_dtg_mpcc_wallsoff",
        demo=RUNS / "f4c_plain_DEMO",
        control=(1.0, RUNS / "f4c_plain_CTRL"),
        paced=[(0.5, RUNS / "f4c_plain_D05"),
               (1.5, RUNS / "f4c_plain_D15"),
               (2.5, RUNS / "f4c_plain_D25")]),
    "Cube": dict(cube=True,
        champion=AGG / "champ_cube_repro",
        demo=OLD / "f4_cube_DEMO_s1warm",
        control=(1.0, OLD / "f4_cube_CTRL_s1warm"),
        paced=[(0.3, RUNS / "f4c_cube_D03"),
               (0.75, RUNS / "f4c_cube_D075"),
               (1.2, RUNS / "f4c_cube_D12")]),
    "Slippery PushT": dict(cube=False,
        champion=FX / "fix_slipCH_current",
        demo=RUNS / "f4c_slippery_DEMO",
        control=(1.0, RUNS / "f4c_slippery_CTRL"),
        paced=[(0.5, RUNS / "f4c_slippery_D05"),
               (1.5, RUNS / "f4c_slippery_D15"),
               (2.5, RUNS / "f4c_slippery_D25")]),
}
PREV = {   # published cells, for the faint comparison curve
    "Plain PushT": dict(demo=FX / "fix_plainDEMO_current", control=(1.0, FX / "fix_plainCTRL_current"),
        paced=[(0.5, FX / "fix_plainD05_current_cref"), (1.5, FX / "fix_plainD15_current_cref"), (2.5, FX / "fix_plainD25_current_cref")]),
    "Cube": dict(demo=FX / "fix_cubeDEMO_current", control=(1.0, FX / "fix_cubeCTRL_current"),
        paced=[(0.3, FX / "fix_cubeD03_current_cref"), (0.75, FX / "fix_cubeD075_current_cref"), (1.2, FX / "fix_cubeD12_current_cref")]),
    "Slippery PushT": dict(demo=WO / "exp6_slippery_demoonly_wallsoff", control=(1.0, FX / "fix_slipCTRLv1_current"),
        paced=[(0.5, B / "logs_epcc5/rfx_slipD05_current_cref"), (1.5, B / "logs_epcc5/rfx_slipD15_current_cref"), (2.5, B / "logs_epcc5/rfx_slipD25_current_cref")]),
}
CORPORA = {
    "Plain PushT": [("champion", H/"datasets/pusht_expert_train.h5"),
                    ("c=0.5", H/"datasets/agg_plain_D_c0.5.h5"),
                    ("c=1.0\n(control)", H/"datasets/pusht_expert_train_sub83k.h5"),
                    ("c=1.5", H/"datasets/agg_plain_D_c1.5.h5"),
                    ("c=2.5", H/"datasets/agg_plain_D_c2.5.h5"),
                    ("demo-only", H/"datasets/expert_slow_025_n200_centered_v2_park.h5")],
    "Slippery PushT": [("champion", H/"slippery_pusht/data/slippery_pusht_wm_corpus.h5"),
                       ("c=0.5", H/"datasets/agg_slippery_D_c0.5.h5"),
                       ("c=1.0\n(control)", H/"slippery_pusht/data/slippery_pusht_wm_corpus_v1_sub101k.h5"),
                       ("c=1.5", H/"datasets/agg_slippery_D_c1.5.h5"),
                       ("c=2.5", H/"datasets/agg_slippery_D_c2.5.h5"),
                       ("demo-only", H/"slippery_pusht/data/expert_slow_050_n200_d035_park.h5")],
}


def score(run, cube=False):
    if not (run / "results.json").is_file():
        return None
    if cube:
        m = pm.cube_ttc(run)
        ok = np.asarray(m["success"], bool); t = np.asarray(m["ttc_s"], float)[ok]
        ct = 0.05
    else:
        m = pm.park_metrics(run, hold_k=HOLD)
        if m is None:
            return None
        ok = np.asarray(m["success"], bool); t = np.asarray(m["ttc_s"], float)[ok]
        ct = float(m["results"].get("control_ts_mean", pm.DT))
    n = int(ok.size)
    tmax = np.load(run / "rollout_traj.npz")["action"].shape[0] * ct
    tpr = (sum(1.0 / max(x, 1e-9) for x in t) - (n - int(ok.sum())) / tmax) / n
    return dict(sr=float(ok.mean()),
                ttc=float(np.median(t)) if t.size else np.nan, tpr=tpr, n=n)


def block_xy(p):
    with h5py.File(p, "r") as f:
        if "state" not in f:
            return None
        st = np.asarray(f["state"][:], np.float64)
    return st[:, 2:4] if st.shape[1] >= 5 else None


def marginals():
    with paper_style():
        rows = list(CORPORA)
        fig, axes = plt.subplots(len(rows), 6, figsize=(DOUBLE_COL, 2.6),
                                 constrained_layout=True)
        for r, task in enumerate(rows):
            xy = {n: block_xy(p) for n, p in CORPORA[task] if p.is_file()}
            allxy = np.concatenate([v for v in xy.values() if v is not None])
            ex = [np.linspace(allxy[:, i].min(), allxy[:, i].max(), NB + 1) for i in (0, 1)]
            hs = {}
            for n, v in xy.items():
                h, _, _ = np.histogram2d(v[:, 0], v[:, 1], bins=ex)
                hs[n] = h / max(h.sum(), 1)
            vmax = max(h.max() for h in hs.values())
            for c, (n, _) in enumerate(CORPORA[task]):
                ax = axes[r][c]
                if n in hs:
                    ax.imshow(hs[n].T, origin="lower", cmap="viridis",
                              vmin=0, vmax=vmax, aspect="auto")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(n, fontsize=7)
                if c == 0:
                    ax.set_ylabel(task.replace(" PushT", ""), fontsize=8)
        fig.suptitle("Where each corpus puts the block  (2D density, shared scale per row)",
                     fontsize=9, fontweight="bold")
    save_fig(fig, OUT / "cgrid_marginals.png")
    print("wrote", OUT / "cgrid_marginals.png", flush=True)


def sweep():
    data = {}
    for task, cfg in CELLS.items():
        cube = cfg["cube"]
        pts = sorted([cfg["control"]] + cfg["paced"], key=lambda t: t[0])
        rows = [(c, score(r, cube)) for c, r in pts]
        data[task] = dict(pts=[(c, s) for c, s in rows if s],
                          ctrl_c=cfg["control"][0],
                          champ=score(cfg["champion"], cube),
                          demo=score(cfg["demo"], cube))
        # the PUBLISHED cells (H16/K2/stride-5 generator, cold ring), drawn faint for comparison
        pcfg = PREV.get(task)
        if pcfg and os.environ.get("SHOW_PREV", "1") == "1":
            ppts = sorted([pcfg["control"]] + pcfg["paced"], key=lambda t: t[0])
            data[task]["prev_pts"] = [(c, s) for c, s in ((c, score(r, cube)) for c, r in ppts) if s]
            data[task]["prev_demo"] = score(pcfg["demo"], cube)
        else:
            data[task]["prev_pts"], data[task]["prev_demo"] = [], None
        print(f"\n{task}")
        for c, s in rows:
            print(f"  c={c:<5} SR {s['sr']:.2f}  TTC {s['ttc']:6.2f}  TPR {s['tpr']:.4f}"
                  if s else f"  c={c:<5} MISSING")
        for k, note in (("champ", "(measured, NOT drawn)"), ("demo", f"(c={DEMO_C[task]})")):
            s = data[task][k]
            print(f"  {k:<9} " + (f"SR {s['sr']:.2f}  TTC {s['ttc']:6.2f}  TPR {s['tpr']:.4f}"
                                  if s else "MISSING") + f"  {note}")

    missing = [t for t, d in data.items() if not d["pts"]]
    if missing:
        print("SKIPPED (no campaign runs yet):", missing)
    data = {t: d for t, d in data.items() if d["pts"]}
    import json as _json
    OUT.mkdir(parents=True, exist_ok=True)
    (FINAL / "scores").mkdir(exist_ok=True)
    (FINAL / "scores/fig4_scores.json").write_text(_json.dumps(
        {t: {"points": [(c, s) for c, s in d["pts"]], "demo": d["demo"], "champion(not drawn)": d["champ"]}
         for t, d in data.items()}, indent=1, default=str))

    # Axes report SR in percent; the scores json above keeps the fraction it was
    # scored in, so the recorded numbers stay comparable with every other script.
    for d in data.values():
        for _s in ([x for _, x in d["pts"]] + [x for _, x in d.get("prev_pts") or []]
                   + [d.get("demo"), d.get("prev_demo"), d.get("champ")]):
            if _s and np.isfinite(_s.get("sr", np.nan)):
                _s["sr"] = 100.0 * _s["sr"]
    with paper_style(), _mpl.rc_context(_RC1TO1):
        COMPACT = _os.environ.get("COMPACT", "0") == "1"
        METRICS = ((("sr", "SR (%)"), ("ttc", "TTC (s)"), ("tpr", "TP (1/s)")) if COMPACT else
                   (("sr", "Success rate (%)"), ("ttc", "TTC (s)"), ("tpr", "TP (1/s)")))
        fig, axes = plt.subplots(len(METRICS), len(data), figsize=(FIG_W * len(data) / 3, FIG_H),
                                 sharex="col", constrained_layout=True, squeeze=False)
        # The rightmost panel's title is wider than its axes ("Slippery-PushT" in the
        # paper's task naming), and constrained_layout sizes columns from the axes, not
        # from the title, so it runs off the canvas. Pull the axes area in from the right
        # edge by RIGHT_RECT; the title then centres inside the figure.
        try:
            fig.get_layout_engine().set(rect=(0.0, 0.0,
                                              float(_os.environ.get("RIGHT_RECT", 1.0)), 1.0))
        except Exception as _e:
            print(f"[layout] rect not applied ({_e}); title may clip", flush=True)
        for j, (task, d) in enumerate(data.items()):
            cs = [xval(task, c) for c, _ in d["pts"]]
            for i, (k, ylab) in enumerate(METRICS):
                ax = axes[i][j]
                ys = [s[k] for _, s in d["pts"]]
                ax.plot(cs, ys, color=series_color(0, 2), lw=1.6, zorder=5,
                        label=("paced play (s1H84 + warm)" if _os.environ.get("SHOW_PREV", "1") == "1" else "paced play") if (i == 0 and j == 0) else None)
                pp = d.get("prev_pts", [])
                if pp:
                    ax.plot([xval(task, c) for c, _ in pp], [s[k] for _, s in pp], color=series_color(1, 2), lw=1.0,
                            ls=":", marker="o", ms=2.2, mfc="white", mec=series_color(1, 2), zorder=4,
                            label="published (H16, cold ring)" if (i == 0 and j == 0) else None)
                pdm = d.get("prev_demo")
                if pdm and np.isfinite(pdm[k]):
                    ax.axhline(pdm[k], ls=":", lw=0.9, color=series_color(1, 2), zorder=2)
                for c, s in d["pts"]:
                    sub = abs(c - d["ctrl_c"]) < 1e-9
                    # FILLED=1: the control point is drawn like every other pace (the
                    # caption no longer singles it out), so no hollow marker.
                    hollow = sub and _os.environ.get("FILLED", "0") != "1"
                    ax.plot([xval(task, c)], [s[k]], marker="o", ms=3.4 if hollow else 2.6,
                            mfc="white" if hollow else series_color(0, 2),
                            mec=series_color(0, 2), mew=1.1, zorder=6, linestyle="none")
                # demo-only spans the panel: it is the level the sweep is measured
                # against, not a competitor at one pace. Grey and behind, per the
                # ours-in-front convention.
                sc = d["demo"]
                if sc and np.isfinite(sc[k]):
                    ax.axhline(sc[k], ls="--", lw=1.1, color=series_color(1, 2),
                               zorder=3,
                               label="Expert-Only" if (i == 0 and j == 0) else None)
                if i == 0:
                    ax.set_title(DISP.get(task, task),
                                 fontsize=PT, fontweight="bold", pad=3)
                    ax.set_ylim(0, 100)
                else:
                    # leave room so the demo-only rule never sits flush against a
                    # spine when it falls outside the swept range
                    ax.margins(y=0.14)
                if j == 0:
                    ax.set_ylabel(ylab, labelpad=1.5)
                if k == "tpr":
                    # TPR is signed: below zero a method is losing more to failures than
                    # it gains from fast successes, so the axis needs its own zero
                    ax.axhline(0, ls="-", lw=0.8, color="0.6", zorder=1)
                if i == len(METRICS) - 1 and (not COMPACT or j == len(data) // 2):
                    ax.set_xlabel("play command magnitude ($\\times$ demo)" if MEASURED_X else "collection pace c")   # one shared label at column width
                sparse_ticks(ax, "y", 3)
                # pace is a multiplicative knob (half speed, double speed), so the axis
                # is log: on a linear axis plain's 0.25 and 0.5 collided while 1.5 and
                # 2.5 sat in empty space
                from matplotlib.ticker import NullLocator
                xt = sorted(set(cs))
                ax.set_xscale("log")
                ax.xaxis.set_minor_locator(NullLocator())
                ax.set_xticks(xt)
                # At column width a panel is ~0.8 in wide while an 8 pt "0.75" is 0.27 in,
                # so every pace cannot carry a label. Keep the ends and any middle tick
                # that clears its neighbours by XTICK_SEP of the log span; the ticks
                # themselves stay, only their labels are thinned.
                import math
                lo, hi = math.log10(min(xt)), math.log10(max(xt))
                sep = float(_os.environ.get("XTICK_SEP", 0.0))
                span = max(hi - lo, 1e-9)
                keep, last = [], None
                for c in xt:
                    q = (math.log10(c) - lo) / span
                    if c == xt[0] or c == xt[-1] or last is None or (q - last) >= sep:
                        if c == xt[-1] and last is not None and (1.0 - last) < sep and len(keep) > 1:
                            keep[-1] = ""          # drop the crowded neighbour, keep the end
                        keep.append(f"{c:.1f}" if MEASURED_X else f"{c:g}"); last = q
                    else:
                        keep.append("")
                if COMPACT:
                    # vertical labels keep every pace legible without eating panel width.
                    # On \Cube the control c=1 sits 0.08 decades from the last pace 1.2,
                    # which is narrower than an 8 pt glyph is tall, so its label is dropped
                    # (the tick and the hollow control marker stay; the caption names it).
                    import math as _m
                    x0, x1 = ax.get_xlim()
                    _span = _m.log10(x1) - _m.log10(x0)
                    _q = [(_m.log10(c) - _m.log10(x0)) / _span for c in xt]
                    _lab = [(f"{c:.1f}" if MEASURED_X else f"{c:g}") for c in xt]
                    _ctrl = xval(task, d["ctrl_c"])   # xt holds plotted abscissae
                    _minsep = float(_os.environ.get("XTICK_MINSEP", 0.15))
                    _last = len(xt) - 1
                    for _i in range(len(xt) - 1):
                        if _q[_i + 1] - _q[_i] < _minsep:
                            # Of a colliding pair, give up the label that costs least:
                            # an interior tick before an end (the ends anchor the range),
                            # and among two interior ticks the one that is not the control
                            # (on \PushT the control is also the best-performing pace, so
                            # dropping it would hide the point the panel is making).
                            _cand = [j for j in (_i, _i + 1) if j not in (0, _last)]
                            if not _cand:
                                _cand = [_i + 1]
                            if len(_cand) > 1:
                                _cand = [j for j in _cand if abs(xt[j] - _ctrl) >= 1e-9] or _cand
                            _lab[_cand[0]] = ""
                    ax.set_xticklabels(_lab, fontsize=PT, rotation=90, va="top", ha="center")
                else:
                    ax.set_xticklabels(keep, fontsize=PT)
                ax.set_xlim(min(xt) / 1.18, max(xt) * 1.18)
        # each row's tick labels have a different width ("0.8" vs "18" vs "0.08"), which
        # otherwise leaves the three y-axis labels at three different x positions
        fig.align_ylabels([axes[i][0] for i in range(len(METRICS))])
        fig.align_xlabels([axes[-1][j] for j in range(len(data))])
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=PT, fancybox=True,
                   edgecolor="0.5", framealpha=0.9, handletextpad=0.4,
                   columnspacing=1.0, borderpad=0.3, handlelength=1.6)
    _out = OUT / f"cgrid_sweep_1to1{_os.environ.get('SUF','')}.png"
    fig.savefig(_out, dpi=400)          # exact canvas, no tight bbox
    print("\nwrote", _out, flush=True)


if __name__ == "__main__":
    # ONLY=sweep skips the marginals panel, which re-reads several multi-GB corpora and
    # dominates the runtime when only the sweep styling has changed.
    only = os.environ.get("ONLY", "")
    if only != "sweep":
        marginals()
    if only != "marginals":
        sweep()
