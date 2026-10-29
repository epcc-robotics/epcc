"""Fast half of Fig. 3b: draw from ../data/alpha_per_episode_tableI.json (dump_fig3b_alpha.py).

No rollout / demo / dataset is opened, so a re-render takes seconds. Aggregation and drawing
mirror contact_split_3task_sailcheck.paired() / main() line for line; the sbatch checks the
PNG + values json byte-for-byte against that script's own outputs.
env: SUP_MODE full|robot  AGG paired4|own  ERR 0|1  CUBE_MAP hold|precision|grasp
     FIG_W FIG_H FIG_PT RIGHT_RECT PHASE_XLAB NOLEG LEG_NCOL SUF   (same meaning as the scorer)
     OUT_DIR (default ../render/tableI[_suprobot][_own])
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as _mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "findings/260907_final_dtg_s1_warm/scripts"))
from paper_style import paper_style, series_color, sparse_ticks  # noqa: E402

SUP_MODE = os.environ.get("SUP_MODE", "full"); assert SUP_MODE in ("full", "robot")
AGG = os.environ.get("AGG", "paired4"); assert AGG in ("paired4", "own")
ERR = os.environ.get("ERR", "0") == "1"
CUBE_MAP = os.environ.get("CUBE_MAP", "hold")
FIG_W = float(os.environ.get("FIG_W", 3.54)); FIG_H = float(os.environ.get("FIG_H", 2.55))
PT = float(os.environ.get("FIG_PT", 8))
_RC1TO1 = {"font.size": PT, "axes.titlesize": PT, "axes.labelsize": PT,
           "xtick.labelsize": PT, "ytick.labelsize": PT, "legend.fontsize": PT,
           "figure.dpi": 400, "savefig.dpi": 400}
OUT = Path(os.environ.get("OUT_DIR", HERE.parent / "render" / ("tableI" + ("_suprobot" if SUP_MODE == "robot" else "")
                                                               + ("_own" if AGG == "own" else ""))))

D = json.load(open(HERE.parent / "data/alpha_per_episode_tableI.json"))
ORDER, GROUPS = D["meta"]["order"], D["meta"]["groups"]
CSET = set(D["meta"]["cube_maps"][CUBE_MAP])
M = D[SUP_MODE]


def push_per(name):
    return {l: {tuple(int(x) for x in k.split(":")): v for k, v in eps.items()} for l, eps in M[name].items()}


def cube_per():
    out = {}
    for l, eps in M["Cube"].items():
        acc = {}
        for ep, row in eps.items():
            tot = {"contact": [0.0, 0.0], "free motion": [0.0, 0.0]}
            for ph, (d, o) in row.items():
                g = "contact" if ph in CSET else "free motion"
                tot[g][0] += d; tot[g][1] += o
            if min(v for gv in tot.values() for v in gv) <= 0:
                continue
            acc[int(ep)] = {g: tot[g][0] / tot[g][1] for g in GROUPS}
        if acc:
            out[l] = acc
    return out


def paired(per):   # == contact_split_3task_sailcheck.paired
    labs = [l for l in ORDER if l in per]
    common = set.intersection(*[set(per[l]) for l in labs]) if labs else set()
    eps = {l: sorted(per[l]) if AGG == "own" else sorted(common) for l in labs}
    res = {l: {g: float(np.median([per[l][e][g] for e in eps[l]])) for g in GROUPS} for l in labs}
    n = {l: len(eps[l]) for l in labs} if AGG == "own" else len(common)
    ci = None
    if ERR:
        rng = np.random.default_rng(0)
        ci = {}
        for l in labs:
            ci[l] = {}
            for g in GROUPS:
                v = np.array([per[l][e][g] for e in eps[l]], float)
                b = np.median(v[rng.integers(0, len(v), (2000, len(v)))], axis=1)
                ci[l][g] = [float(x) for x in np.percentile(b, [2.5, 97.5])]
    return res, n, labs, ci


panels = [("PushT", *paired(push_per("PushT"))), ("Cube", *paired(cube_per())),
          ("Slippery-PushT", *paired(push_per("Slippery-PushT")))]
for name, res, ncom, labs, ci in panels:
    print(f"  {name:15s} n={ncom}")
    for l in labs:
        print(f"    {l:<5} " + "   ".join(
            f"{g} {res[l][g]:.2f}" + (f" [{ci[l][g][0]:.2f},{ci[l][g][1]:.2f}]" if ci else "") for g in GROUPS))

with paper_style(), _mpl.rc_context(_RC1TO1):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0.0, 0.0, float(os.environ.get("RIGHT_RECT", 1.0)), 1.0))
    nm = len(ORDER)
    for ax, (name, res, ncom, labs, ci) in zip(axes, panels):
        x = np.arange(len(GROUPS)); w = 0.8 / nm
        for j, l in enumerate(ORDER):
            if l not in res:
                continue
            xs, vals = x + (j - (nm - 1) / 2) * w, [res[l][g] for g in GROUPS]
            ax.bar(xs, vals, w, color=series_color(j, nm), edgecolor="black", linewidth=0.5,
                   zorder=3, label=l if name == panels[0][0] else None)
            if ci is not None:
                ax.errorbar(xs, vals, yerr=[[v - ci[l][g][0] for v, g in zip(vals, GROUPS)],
                                            [ci[l][g][1] - v for v, g in zip(vals, GROUPS)]],
                            fmt="none", ecolor="black", elinewidth=0.6, capsize=1.2,
                            capthick=0.6, zorder=4)
        ax.axhline(1.0, ls="--", color="0.45", lw=0.8, zorder=2)
        ax.set_xticks(x, ["contact", "free"], fontsize=PT)
        ax.set_xlim(-0.45, 1.45)
        ax.tick_params(pad=1.5)
        ax.set_title(name, fontsize=PT, fontweight="bold", pad=3)
        top = max(v for m in res.values() for v in m.values())
        if ci is not None:
            top = max(top, max(c[1] for m in ci.values() for c in m.values()))
        ax.set_ylim(0, top * 1.12)
        sparse_ticks(ax, "y", 4)
    axes[0].set_ylabel(r"speed-up $\alpha$", labelpad=1.5)
    if os.environ.get("PHASE_XLAB", ""):
        axes[1].set_xlabel(os.environ["PHASE_XLAB"])
    h, lab = axes[0].get_legend_handles_labels()
    if os.environ.get("NOLEG", "0") != "1":
        fig.legend(h, lab, loc="outside lower center", ncol=int(os.environ.get("LEG_NCOL", 4)), fontsize=PT,
                   fancybox=True, edgecolor="0.5", framealpha=0.9, handletextpad=0.3, columnspacing=0.6,
                   borderpad=0.25, handlelength=1.2)
suffix = "" if CUBE_MAP == "hold" else f"_{CUBE_MAP}"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"contact_split_3task{suffix}_values.json").write_text(json.dumps(
    [{"panel": n, ("paired_n" if AGG == "paired4" else "n"): ncom, "alpha": res,
      **({"ci95": ci} if ci is not None else {})} for n, res, ncom, labs, ci in panels], indent=1))
png = OUT / f"contact_split_3task{suffix}_1to1{os.environ.get('SUF', '')}.png"
fig.savefig(png, dpi=400)
print("wrote", png)
