"""Shared paper plotting style for Ablation figures (docs/plot_style.md).

Fixed structure: our method = series index 0, and it is the ONLY saturated colour
on the figure; baselines = index 1..N drawn on a dark-to-light grey ramp, thinner,
behind. The reader should find our series without consulting the legend. Bars keep
hatches as a second channel; lines rely on the grey ramp plus marker fill.
Apply via ``with paper_style():`` around figure building — never mutate global rcParams.

Usage:
    from paper_style import (paper_style, save_fig, series_color, series_hatch,
                             series_linestyle, series_marker,
                             SINGLE_COL, DOUBLE_COL)
    with paper_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
        ...
    save_fig(fig, "out.png")

Shared decisions live here; per-chart decisions (bar width, headroom
override, legend placement) live in the renderer.
"""
from contextlib import contextmanager

import matplotlib as mpl

# ---------------------------------------------------------------- layout ---
SINGLE_COL = 3.35   # inches — author at this width, place at 1:1
DOUBLE_COL = 6.9

# ----------------------------------------------------- series identity -----
# One colour carries meaning: ours. Baselines are a grey ramp, so the eye lands on
# our series first and the ordering survives greyscale printing. A rainbow of
# equally-saturated baselines (the v0-v2 Okabe-Ito palette) made every series shout
# equally, which is the opposite of what a comparison figure needs.
OURS = "#0072B2"                      # Okabe-Ito blue, kept for continuity
GREYS = ["#3D3D3D", "#6E6E6E", "#9A9A9A", "#C2C2C2", "#D9D9D9"]
SERIES = [OURS] + GREYS               # drop-in for series_color(i)

# Kept so a figure that genuinely needs categorical hues (e.g. a per-task panel
# where no series is "ours") can opt in explicitly. Do not use for baselines.
CATEGORICAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
               "#E69F00", "#56B4E9", "#F0E442", "#999999"]

# Second channel (double encoding), paired 1:1 with SERIES by index.
HATCHES = [None, "//", "xx", "\\\\", "..", "++", "oo", "**"]
LINESTYLES = ["-", "--", "-.", ":",
              (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (3, 1))]
MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*"]


def series_color(i, n=None):
    """Colour for series ``i``. ``n`` = total series count; when given, the grey
    ramp is spread evenly over the baselines instead of taking the first few
    entries, which keeps 2-baseline and 5-baseline figures equally legible."""
    if i == 0:
        return OURS
    if n is None or n <= 2:
        return GREYS[(i - 1) % len(GREYS)]
    lo, hi = 0.24, 0.78                      # dark -> light, both print-safe
    t = lo + (hi - lo) * ((i - 1) / max(n - 2, 1))
    return mpl.colors.to_hex((t, t, t))


def series_lw(i):
    """Ours is drawn heavier so it reads as the foreground series."""
    return 2.0 if i == 0 else 1.3


def series_zorder(i):
    """Ours on top; baselines behind, in order."""
    return 5 if i == 0 else 3


def series_fill(i):
    """Marker face: ours filled, baselines hollow (a greyscale-safe channel)."""
    return series_color(i) if i == 0 else "white"


def series_hatch(i):
    return HATCHES[i % len(HATCHES)]


def series_linestyle(i):
    return LINESTYLES[i % len(LINESTYLES)]


def series_marker(i):
    return MARKERS[i % len(MARKERS)]


# --------------------------------------------------------------- rc spec ---
_RC = {
    # typography — DejaVu Sans Mono ships with matplotlib, so this stays
    # reproducible on any machine that can import matplotlib at all.
    "font.family": "monospace",
    "font.monospace": ["DejaVu Sans Mono"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.labelweight": "normal",   # mono at bold is heavy; titles carry the weight
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    # axes and ticks
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # grid (y-axis only, dashed, subtle)
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    # bars / hatches
    "hatch.linewidth": 0.6,
    # legend
    "legend.fancybox": True,
    "legend.edgecolor": "0.5",
    "legend.framealpha": 0.9,
    # layout / output
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
}


@contextmanager
def paper_style():
    """Context-managed rc settings — no global rcParams mutation."""
    with mpl.rc_context(_RC):
        yield


def sparse_ticks(ax, axis="y", n=3, integer=False):
    """Few, round tick values. Matplotlib's default density crowds a column-width
    panel; the paper's reference figures label 2-4 values per axis and nothing more."""
    loc = mpl.ticker.MaxNLocator(nbins=n, integer=integer, prune=None)
    if axis in ("y", "both"):
        ax.yaxis.set_major_locator(loc)
    if axis in ("x", "both"):
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=n, integer=integer))
    return ax


def save_fig(fig, path):
    """Save at publication settings: 300 dpi, tight bbox, 0.02" pad."""
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)


# ----------------------------------------------------------- style stack ---
# Changelog of style decisions, newest last.
#
# - v0 - baseline: palette, typography, spines/grid, widths, 300 dpi.
# - v1 - hatch + edges:
#     shared: + HATCHES, + series_hatch(i), + hatch.linewidth = 0.6
#     bars (renderer): + hatch=series_hatch(i), + edgecolor="black",
#                      linewidth=0.5
#     headroom (renderer): ylim * 1.12 (* 1.35 on legend collision)
# - v2 - module creation (2026-07-11): consolidate v0+v1 into this shared
#   module for Ablation A paper figures.
#     shared: + paper_style() via mpl.rc_context (no global mutation),
#             + SERIES / HATCHES / LINESTYLES / MARKERS,
#             + series_color/series_hatch/series_linestyle/series_marker,
#             + SINGLE_COL=3.35 / DOUBLE_COL=6.9,
#             + save_fig(fig, path) (dpi=300, bbox_inches="tight",
#               pad_inches=0.02),
#             + rc: axes.grid=y-only dashed alpha .3, constrained_layout on
#     chart-specific: none (bar width, headroom, legends stay in renderers)
# - v3 - foreground/background split (2026-08-27): make ours the only saturated
#   series so it is findable without the legend; match the reference figures the
#   group settled on (mono type, sparse ticks).
#     shared: SERIES  -> Okabe-Ito rainbow -> OURS + GREYS ramp
#             + OURS / GREYS / CATEGORICAL (old palette kept, opt-in only)
#             series_color(i) -> series_color(i, n=None)  (ramp spreads over n)
#             + series_lw(i)      ours 2.0 / baselines 1.3
#             + series_zorder(i)  ours 5 / baselines 3
#             + series_fill(i)    ours filled / baselines hollow markers
#             + sparse_ticks(ax, axis, n)  MaxNLocator, 3 bins by default
#             font.family      -> sans-serif -> monospace (DejaVu Sans Mono)
#             axes.labelweight -> bold -> normal
#     chart-specific: lines use the grey ramp + marker fill and DROP linestyle
#             cycling (the ramp already separates them); bars keep hatch.
