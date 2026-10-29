"""Render the equal-time results from the score json files (no rollout is opened).

in : ../scores/table1_eqtime.json, ../scores/fig3a_eqtime_{published,robot}.json,
     ../render/fig3b/<variant>/contact_split_3task_precision_values.json,
     published Fig. 3b values (findings/261022_fig3b_sail/render/<variant>/...)
out: ../render/table1/table1_eqtime.md   (Table I: published vs equal-time, both SuP variants)
     ../render/summary.md                (derived paper numbers, Fig. 3a / 3b at equal time)
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = Path(__file__).resolve().parents[3]
T1 = json.loads((ROOT / "scores/table1_eqtime.json").read_text())
TASKS = [("pusht", "PushT"), ("cube", "Cube"), ("slippery", "Slippery-PushT")]
ROWS = ["DP", "SkiP", "SAIL", "SuP", "SuP_robot", "PlayGoal", "EPCC"]
ACCEL = ["SkiP", "SAIL"]          # + the SuP variant


def f3(r):
    ttc = "–" if r["TTC_median_s"] is None or math.isnan(r["TTC_median_s"]) else f"{r['TTC_median_s']:.1f}"
    tp = f"{r['TP']:.3f}".replace("0.", ".", 1) if r["TP"] >= 0 else f"−{abs(r['TP']):.3f}".replace("0.", ".", 1)
    return f"{r['SR_pct']:.0f} / {ttc} / {tp}"


L = [f"# Table I at equal simulated time per task (2026-10-25)", "",
     f"T = {T1['T_task_s']} s. SR % / TTC s / TP. 'published' = the run's own length and t_max "
     f"(reproduced exactly, regression failures: {T1['regression_failures']}); 'equal time' = cut at T, t_max = T.", "",
     "| task | row | published | equal time | successes lost by the cut | run |", "|---|---|---|---|---|---|"]
for t, tn in TASKS:
    for r in ROWS:
        x = T1["rows"][f"{t}/{r}"]
        L.append(f"| {tn} | {r} | {f3(x['own'])} | {f3(x['eq'])} | {x['lost_by_cut']} | `{x['run']}` |")
L += ["", "## Ratios (TP)", "", "| task | SuP variant | EPCC / DP (pub → eq) | EPCC / best acceleration baseline (pub → eq) |",
      "|---|---|---|---|"]
S = ["# Equal-time rescoring — derived numbers (2026-10-25)", ""]
for t, tn in TASKS:
    for sv in ("SuP", "SuP_robot"):
        cells = []
        for col in ("own", "eq"):
            g = {r: T1["rows"][f"{t}/{r}"][col]["TP"] for r in ROWS}
            best = max(ACCEL + [sv], key=lambda r: g[r])
            cells.append((g["EPCC"] / g["DP"], g["EPCC"] / g[best], best))
        L.append(f"| {tn} | {sv} | {cells[0][0]:.2f}× → {cells[1][0]:.2f}× | "
                 f"{cells[0][1]:.2f}× ({cells[0][2]}) → {cells[1][1]:.2f}× ({cells[1][2]}) |")
(ROOT / "render/table1").mkdir(parents=True, exist_ok=True)
(ROOT / "render/table1/table1_eqtime.md").write_text("\n".join(L) + "\n")
S += L[L.index("## Ratios (TP)"):] + [""]

# Fig. 3a
DS = ["d000", "d010", "d020", "d030", "d040", "d050", "d060", "d070", "d080"]
TAU = [0.0] + [1.0 / math.log(1.0 / (int(d[1:]) / 100)) for d in DS[1:]]
for m in ("published", "robot"):
    pub = json.loads((ROOT / f"scores/regress/fig3a_{m}.json").read_text())
    eq = json.loads((ROOT / f"scores/fig3a_eqtime_{m}.json").read_text())
    S += [f"## Fig. 3a (SuP = {m}) — SR % / TP, published → equal time (59.4 s)", "",
          "| τ (s) | " + " | ".join(eq) + " |", "|---|" + "---|" * len(eq)]
    for i, d in enumerate(DS):
        row = []
        for lab in eq:
            a, b = pub[lab][d], eq[lab][d]
            row.append(f"{100*float(a['hold']):.0f}/{float(a['tpr']):+.3f} → {100*float(b['hold']):.0f}/{float(b['tpr']):+.3f}")
        S.append(f"| {TAU[i]:.2f} | " + " | ".join(row) + " |")
    S.append("")

# Fig. 3b
for var in ("tableI", "tableI_own", "tableI_suprobot", "tableI_suprobot_own", "tableI_suprobot_own_noci"):
    fn = "contact_split_3task_precision_values.json"
    pub = json.loads((REPO / "findings/261022_fig3b_sail/render" / var / fn).read_text())
    eq = json.loads((ROOT / "render/fig3b" / var / fn).read_text())
    S += [f"## Fig. 3b `{var}` — α contact / free, published → equal time", "",
          "| panel | n (pub → eq) | " + " | ".join(pub[0]["alpha"]) + " |", "|---|---|" + "---|" * len(pub[0]["alpha"])]
    derived = {}
    for p, q in zip(pub, eq):
        nk = "paired_n" if "paired_n" in p else "n"
        cells = [f"{p['alpha'][l]['contact']:.2f}/{p['alpha'][l]['free motion']:.2f} → "
                 f"{q['alpha'][l]['contact']:.2f}/{q['alpha'][l]['free motion']:.2f}" for l in p["alpha"]]
        S.append(f"| {p['panel']} | {p[nk]} → {q[nk]} | " + " | ".join(cells) + " |")
    for tag, vals in (("published", pub), ("equal time", eq)):
        r = [v["alpha"]["EPCC"]["contact"] / max(a["contact"] for l, a in v["alpha"].items() if l != "EPCC") for v in vals]
        mc = lambda l, g: sum(v["alpha"][l][g] for v in vals) / len(vals)  # noqa: E731
        derived[tag] = (sum(r) / len(r), mc("EPCC", "contact"), mc("SuP", "contact"), mc("EPCC", "free motion"), mc("SuP", "free motion"))
    S += ["", "| | contact α over strongest baseline (mean of tasks) | contact EPCC vs SuP | free EPCC vs SuP |", "|---|---|---|---|"]
    for tag, d in derived.items():
        S.append(f"| {tag} | {d[0]:.2f}× | {d[1]:.2f} vs {d[2]:.2f} | {d[3]:.2f} vs {d[4]:.2f} |")
    S.append("")
(ROOT / "render/summary.md").write_text("\n".join(S) + "\n")
print("\n".join(L)); print(); print("\n".join(S))
