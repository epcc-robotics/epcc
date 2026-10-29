"""Before/after the budget consistency fix (2026-10-25). SuP = robot-only in both,
so the only difference is the evaluation time. Reads the score json files of this tree (no rollout).

in : ../scores/table1_eqtime.json (own = before, eq = after); Fig. 3a / 3b PNGs already rendered:
     before = ../render/regress/... (own budgets, byte-identical to the published renders),
     after  = ../render/fig3a/robot, ../render/fig3b/tableI_suprobot_own_noci
out: ../render/before_after/{table1_before_after.png, table1_before_after.md,
                             fig3a_before_after.png, fig3b_before_after.png}
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "render/before_after"
OUT.mkdir(parents=True, exist_ok=True)
D = json.loads((ROOT / "scores/table1_eqtime.json").read_text())
R, T = D["rows"], D["T_task_s"]
TASKS = [("pusht", "PushT"), ("cube", "Cube"), ("slippery", "Slippery-PushT")]
ROWS = [("DP", "DP"), ("SkiP", "SkiP"), ("SAIL", "SAIL"), ("SuP_robot", "SuP"), ("PlayGoal", "Play+Goal"), ("EPCC", "EPCC")]


def f(m, key):
    v = m[key]
    if key == "SR_pct":
        return f"{v:.0f}"
    if key == "TTC_median_s":
        return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"
    s = f"{v:.3f}"
    return s.replace("0.", ".", 1) if v >= 0 else "-" + s[1:].replace("0.", ".", 1)


KEYS = ["SR_pct", "TTC_median_s", "TP"]
cells, changed, md = [], [], []
SHORT = {"PushT": "PushT", "Cube": "Cube", "Slippery-PushT": "Slippery"}
hdr = ["Method"] + [f"{SHORT[tn]}\n{k}" for _, tn in TASKS for k in ("SR %", "TTC s", "TP")]
md.append("| Method | " + " | ".join(f"{tn} {k}" for _, tn in TASKS for k in ("SR %", "TTC s", "TP")) + " |")
md.append("|---|" + "---|" * 9)
for key, lab in ROWS:
    row, ch, mrow = [lab], [False], [lab]
    for t, _ in TASKS:
        x = R[f"{t}/{key}"]
        for k in KEYS:
            a, b = f(x["own"], k), f(x["eq"], k)
            row.append(a if a == b else f"{a} → {b}"); ch.append(a != b)
            mrow.append(a if a == b else f"**{a} → {b}**")
    cells.append(row); changed.append(ch); md.append("| " + " | ".join(mrow) + " |")

# evaluation time per row, before -> after
tm = [["t_max (s)"]]
for t, tn in TASKS:
    own = sorted({round(R[f"{t}/{k}"]["own"]["t_max_s"], 1) for k, _ in ROWS})
    tm[0].append(f"{tn}: " + " / ".join(f"{v:g}" for v in own) + f"  →  {T[t]:g} for all")
md += ["", "Evaluation time t_max (s), before → after: " + "; ".join(tm[0][1:]),
       "", "Before: each method's own budget (PushT: EPCC / Play+Goal 43.5, DP / SkiP / SuP 48.9; SAIL: step budget ×k at "
       "a variable control period, mean 61.9 / 16.3 / 67.3 s, and Cube SAIL's TP used t_max = 47.1 s). After: every rollout "
       "cut at the same simulated time per task, t_max = that time. SuP = robot-only in both. TTC shown to 2 decimals so "
       "the half-second-level changes are visible; TP to 3."]
(OUT / "table1_before_after.md").write_text("# Table I: before / after the budget consistency fix\n\n" + "\n".join(md) + "\n")

fig, ax = plt.subplots(figsize=(13.0, 3.1))
ax.axis("off")
tab = ax.table(cellText=cells, colLabels=hdr, loc="center", cellLoc="center")
tab.auto_set_font_size(False); tab.set_fontsize(8.5); tab.scale(1, 1.55)
for (r, c), cell in tab.get_celld().items():
    if r == 0:
        cell.set_facecolor("#e8eef7"); cell.set_text_props(weight="bold")
    elif changed[r - 1][c]:
        cell.set_facecolor("#fff2b3")
ax.set_title("Table I before → after the budget consistency fix (changed cells highlighted; SuP = robot-only in both)",
             fontsize=10, pad=4)
fig.text(0.01, 0.02, "t_max before: PushT EPCC / Play+Goal 43.5 s, DP / SkiP / SuP 48.9 s, SAIL 61.9 s (mean, variable rate); "
         "Cube 15.7 s, SAIL 16.3 s (TP had used 47.1 s); Slippery 59.4 s, SAIL 67.3 s.   "
         "After: 43.5 / 15.7 / 59.4 s for every method.", fontsize=7.5)
fig.savefig(OUT / "table1_before_after.png", dpi=220, bbox_inches="tight")
plt.close(fig)


def side(before, after, title, dst, after_lab):
    a, b = mpimg.imread(before), mpimg.imread(after)
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 3.4))
    for ax, im, t in zip(axs, (a, b), ("Before: each method's own budget", after_lab)):
        ax.imshow(im); ax.axis("off"); ax.set_title(t, fontsize=10)
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.savefig(dst, dpi=200, bbox_inches="tight")
    plt.close(fig)


side(ROOT / "render/regress/fig3a/robot/friction_plain3way_1to1_named.png",
     ROOT / "render/fig3a/robot/friction_plain3way_1to1_named.png",
     "Fig. 3a (SuP robot-only). Only SAIL changes: it ran 1188 steps at a variable control period (59.4-97 s)",
     OUT / "fig3a_before_after.png", "After: every method cut at 59.4 s")
side(ROOT / "render/regress/fig3b/tableI_suprobot_own_noci/contact_split_3task_precision_1to1_prec.png",
     ROOT / "render/fig3b/tableI_suprobot_own_noci/contact_split_3task_precision_1to1_prec.png",
     "Fig. 3b (own-episode medians, SuP robot-only). Only episodes that no longer finish in time drop (SAIL, SkiP)",
     OUT / "fig3b_before_after.png", "After: equal time (43.5 / 15.7 / 59.4 s)")
print((OUT / "table1_before_after.md").read_text())
print("wrote", sorted(p.name for p in OUT.iterdir()))
