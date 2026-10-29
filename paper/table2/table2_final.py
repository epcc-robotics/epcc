"""Table II (SELECT / EPCC(seeded) / EPCC) from the campaign runs, scored with the paper
yardstick via score_runs.score (park_metrics/cube_ttc). Writes $EPCC_PAPER_OUT/table2/table2_scores.json.

  python table2_final.py            (compute node: numpy)
"""
import json, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/common"))
from score_runs import score  # noqa

# Published runs (EPCC_PUBLISHED: a tree with the findings/<campaign>/ layout); outputs to EPCC_PAPER_OUT/table2.
FINAL = Path(os.environ.get("EPCC_PUBLISHED", REPO)) / "findings/260907_final_dtg_s1_warm"
RUNS = FINAL / "runs/table2"
OUT = Path(os.environ.get("EPCC_PAPER_OUT", REPO / "results" / "paper")) / "table2"
# Table II of the paper (SR % / TTC s / TP 1/s, as printed).
PUB = {"plain": {"select": (92, 12.6, .069), "fullcand": (94, 10.0, .092), "full": (78, 10.4, .073)},
       "cube":  {"select": (96, 3.5, .270), "fullcand": (78, 2.6, .281), "full": (84, 2.5, .317)},
       "slip":  {"select": (16, 24.5, -.005), "fullcand": (60, 20.2, .030), "full": (62, 15.8, .033)}}
LABEL = {"select": "SELECT", "fullcand": "EPCC (seeded)", "full": "EPCC"}
rows = {}
print(f"{'row':<14} {'PushT':>22} {'Cube':>22} {'Slippery':>22}")
for rung in ("select", "fullcand", "full"):
    cells = []
    for task in ("plain", "cube", "slip"):
        d = RUNS / f"t2_{task}_{rung}_s1warm"
        r = score(d) if (d / "results.json").is_file() else {"error": "missing"}
        rows[f"{task}/{rung}"] = r
        if "error" in r:
            cells.append(f"{'—':>22}")
        else:
            p = PUB[task][rung]
            cells.append(f"{r['SR_pct']:4.0f}/{r['TTC_median_s']:5.2f}/{r['TP']:+.3f} (pub {p[0]}/{p[1]}/{p[2]:+.3f})")
    print(f"{LABEL[rung]:<14} " + " ".join(cells))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "table2_scores.json").write_text(json.dumps({"published": PUB, "rows": rows}, indent=1))
print("wrote", OUT / "table2_scores.json")
