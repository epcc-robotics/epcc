
# Plot Style

**Module:** `scripts/baseline/paper_style.py`
(`Ablation/A_dist_shift/paper_style.py` is a symlink to it — one file, no sync needed.)

```python
import sys; sys.path.insert(0, "scripts/baseline")
from paper_style import (paper_style, save_fig, sparse_ticks,
                         series_color, series_lw, series_zorder, series_fill,
                         series_hatch, SINGLE_COL, DOUBLE_COL)

n = len(methods)                       # ours + baselines
with paper_style():
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.1), constrained_layout=True)
    for i, m in enumerate(methods):    # i == 0 MUST be ours
        ax.plot(x, y[i], marker="o", ms=3.5, label=m,
                color=series_color(i, n), lw=series_lw(i), zorder=series_zorder(i),
                mfc=series_fill(i), mec=series_color(i, n), mew=1.1)
    sparse_ticks(ax, "y", 3)
save_fig(fig, out_path)
```

Figures already on this style: `scripts/analysis/plot_agg_mainbench_{plain,slip}_hold50.py`,
`plot_friction_sweep.py`, `plot_guard_review.py`, `plot_agg_cube_matrix.py`. Copy their
structure rather than restyling from scratch.

🔴 This project forbids matplotlib/numpy imports on the login node. Render via `sbatch`
(`scripts/analysis/run_analysis.sbatch`), never inline.

## Fixed structure: ours in front, baselines behind

Every comparison is our method vs `N` baselines. Encode it; do not reinvent it per figure.

- **Ours = series index `0`.** The only saturated colour on the figure (`#0072B2`).
  Heaviest line, filled markers, drawn on top. A reader must be able to find it without
  reading the legend.
- **Baselines = index `1..N`.** A dark-to-light grey ramp, thinner, behind, hollow markers.
  They are context, not competitors for attention.
- Ordering must survive greyscale printing — the ramp does this by construction.

Do not give a baseline its own hue to "make it stand out". If one baseline matters more
than the others, say so in the text or annotate it; do not break the ramp.

## Concrete style spec

### Layout and output

- Widths: `SINGLE_COL = 3.35"`, `DOUBLE_COL = 6.9"`. Author at these widths, place at 1:1.
- `constrained_layout`. Save with `dpi=300`, `bbox_inches="tight"`, `pad_inches=0.02`.

### Typography

DejaVu Sans **Mono** (ships with matplotlib, so it renders identically everywhere):

- Base `9` · Title `10` bold · Axis label `9` **normal** (mono at bold is too heavy;
  the title carries the weight) · Ticks `8` · Legend `8`

### Colour

```
OURS   = "#0072B2"
GREYS  = ["#3D3D3D", "#6E6E6E", "#9A9A9A", "#C2C2C2", "#D9D9D9"]
```
`series_color(i, n)` returns ours for `i == 0` and spreads the ramp over `[0.24, 0.78]`
lightness for the baselines, so a 2-baseline and a 5-baseline figure are equally legible.
Pass `n` whenever you know the series count.

`CATEGORICAL` (the old Okabe-Ito rainbow) is still exported for the rare panel where no
series is "ours" — e.g. per-task colours. Never use it for baselines.

### Second channel

- **Bars** — `HATCHES = [None, "//", "xx", "\\\\", "..", "++", "oo", "**"]`, paired 1:1
  with the series index. `hatch.linewidth = 0.6`; every bar gets `edgecolor="black"`,
  `linewidth=0.5`.
- **Lines** — the grey ramp plus marker fill (`series_fill`) already separate the series.
  Do **not** also cycle linestyle; dashed greys on a small panel read as noise.

### Axes and ticks

- `axes.linewidth = 0.8`; ticks `direction="out"`, `width=0.8`, `length=3.5`
- Top and right spines off
- Grid: dashed, `alpha=0.3`, `linewidth=0.6`, y-axis only
- **Tick density: `sparse_ticks(ax, axis, n=3)`.** Matplotlib's default crowds a
  column-width panel. Label 2-4 values per axis and nothing more. For categorical x
  (dataset sizes, phase names) set the ticks explicitly instead.

### Legend

`fancybox=True`, `edgecolor="0.5"`, `framealpha=0.9`. Ours first in the legend, matching
series order.

### Headroom

`ylim(0, max * 1.12)`; use `* 1.35` when a bar would collide with an in-axes legend.

## Rules

- Point-sized, 1:1. Fonts in points, figures at column width; never inflate for downscaling.
- Ours is index 0 in every figure, always.
- Context manager only: `with paper_style():`. No global rcParams mutation.
- Shared decisions (palette, ramp, rc defaults, widths) go in the module; per-chart
  decisions (bar width, headroom override, legend placement) stay in the renderer.
- No chartjunk, decorative colour, or annotations unless requested.
- Any change to the spec goes in the changelog below AND in the module's changelog comment.

## Style stack (changelog)

- `v0` — baseline: palette, typography, spines/grid, widths, 300 dpi.
- `v1` — hatch + edges:
  - shared: `+ HATCHES`, `+ series_hatch(i)`, `+ hatch.linewidth = 0.6`
  - bars: `+ hatch`, `+ edgecolor="black"`, `linewidth=0.5`
  - headroom: `ylim * 1.12` (`* 1.35` on legend collision)
- `v2` — module creation (2026-07-11): consolidate v0+v1 into
  `scripts/baseline/paper_style.py`; `paper_style()` via `mpl.rc_context`; `save_fig`.
- `v3` — foreground/background split (2026-08-27). Intent: make ours findable without the
  legend, and match the reference figures the group settled on.
  - shared: `SERIES` Okabe-Ito rainbow → `OURS + GREYS` ramp ·
    `+ OURS / GREYS / CATEGORICAL` · `series_color(i)` → `series_color(i, n=None)` ·
    `+ series_lw` (2.0 / 1.3) · `+ series_zorder` (5 / 3) · `+ series_fill`
    (filled / hollow) · `+ sparse_ticks(ax, axis, n)` ·
    `font.family` sans-serif → monospace (DejaVu Sans Mono) ·
    `axes.labelweight` bold → normal
  - chart-specific: lines drop linestyle cycling (ramp + fill suffice); bars keep hatch.
