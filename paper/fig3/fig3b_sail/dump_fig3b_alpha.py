"""Slow half of Fig. 3b: extract per-episode alpha ONCE and save it; render_fig3b.py draws from it.

Per-episode values come from contact_split_3task_sailcheck (the figure's own scorer, SAIL_MODE=tableI),
for SuP = full (published) and robot (261023_sup_robot). Saved per task and method:
  PushT / Slippery-PushT : {episode "ep:start": {"contact": a, "free motion": a}}
  Cube                   : raw phase times {episode: {phase: [demo s, method s]}}, so every
                           CUBE_MAPS mapping can be regrouped at render time
EPCC runs come from env EPCC_PUSHT / EPCC_CUBE / EPCC_SLIP (as in score_all.sbatch).
The cube helper rewrites meta.yaml in each cube run dir; md5 is checked before/after.
Writes ../data/alpha_per_episode_tableI.json.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["SAIL_MODE"] = "tableI"
os.environ["SUP_MODE"] = "full"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contact_split_3task_sailcheck as cs  # noqa: E402

REPO = cs.REPO
DST = HERE.parent / "data/alpha_per_episode_tableI.json"


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None


def runs_of(methods, task):
    return {n: [str(r.relative_to(REPO)) for r in rl] for n, rl in cs.pick(methods, task)}


out = {"meta": {"scorer": "findings/261022_fig3b_sail/scripts/contact_split_3task_sailcheck.py",
                "SAIL_MODE": "tableI", "groups": cs.GROUPS, "order": cs.ORDER, "cube_phases": cs.P,
                "cube_maps": {k: sorted(v) for k, v in cs.CUBE_MAPS.items()},
                "EPCC_env": {k: os.environ.get(k) for k in ("EPCC_PUSHT", "EPCC_CUBE", "EPCC_SLIP")}}}
bad = 0
for mode in ("full", "robot"):
    cs.SUP_MODE = mode
    print(f"\n==================== SuP = {mode}", flush=True)
    d = {"runs": {"PushT": runs_of(cs.TASKS["plain"]["methods"], "plain"),
                  "Slippery-PushT": runs_of(cs.TASKS["slippery"]["methods"], "slippery"),
                  "Cube": runs_of(cs.CUBE_METHODS, "cube")}}
    for name, task in (("PushT", "plain"), ("Slippery-PushT", "slippery")):
        per = cs.pusht_alpha(task)
        d[name] = {l: {f"{e[0]}:{e[1]}": v for e, v in sorted(eps.items())} for l, eps in per.items()}
    cube_runs = [REPO / r for rl in d["runs"]["Cube"].values() for r in rl]
    before = {r: md5(r / "meta.yaml") for r in cube_runs}
    pt = cs.cube_phase_times()
    for r in cube_runs:
        if before[r] is not None and md5(r / "meta.yaml") != before[r]:
            print(f"  !! meta.yaml CHANGED: {r}"); bad += 1
    d["Cube"] = {l: {str(e): {ph: list(v) for ph, v in row.items()} for e, row in sorted(eps.items())}
                 for l, eps in pt.items()}
    out[mode] = d
DST.parent.mkdir(exist_ok=True)
DST.write_text(json.dumps(out) + "\n")
print("\nwrote", DST, f"({DST.stat().st_size / 1e3:.0f} kB)", "\nmeta.yaml untouched:", "YES" if not bad else "NO")
sys.exit(1 if bad else 0)
