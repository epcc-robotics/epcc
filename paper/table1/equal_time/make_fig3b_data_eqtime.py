"""Fig. 3b at equal time: drop, from the published per-episode alpha dump, every episode that is
no longer a success once its run is cut at T_TASK. Alpha values themselves do not change: a kept
success completes (park / first reach) before T, and alpha only uses the rollout up to that point.

in : findings/261022_fig3b_sail/data/alpha_per_episode_tableI.json   (read only)
out: ../data/alpha_per_episode_tableI_eqtime.json  (same schema)  +  ../scores/fig3b_filter_log.json
Fails if any dumped episode is not a success at full length under eqtime_common's scorer (that
would mean the two scorers disagree, and the filter would be meaningless).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eqtime_common import REPO, ROOT, T_TASK, cube_metrics, cube_reference, cut, push_metrics  # noqa: E402

SRC = REPO / "findings/261022_fig3b_sail/data/alpha_per_episode_tableI.json"
D = json.loads(SRC.read_text())
PANEL_TASK = {"PushT": "pusht", "Slippery-PushT": "slippery", "Cube": "cube"}
log, bad = {}, 0
cache = {}


def status(run, task):
    """{key: (ok_full, ok_eq)} keyed like the dump: 'ep:st' (push) / 'ep' (cube)."""
    if (run, task) in cache:
        return cache[(run, task)]
    p = REPO / run
    if task == "cube":
        full, eq = cube_metrics(p), cube_metrics(p, T=T_TASK[task])
        ref_ok, _ = cube_reference(p)
        assert np.array_equal(full["ok"], ref_ok), f"GTR recompute != eval success_full: {run}"
        keys = [str(int(e)) for e in full["ep"]]
        okf, oke = full["ok"], eq["ok"]
    else:
        m = push_metrics(p)
        keys = [f"{int(e)}:{int(s)}" for e, s in zip(m["ep"], m["st"])]
        okf, oke = m["ok"], cut(m["ok"], m["ttc"], T_TASK[task])
    st = {}
    for k, a, b in zip(keys, okf, oke):
        if k in st and st[k] != (bool(a), bool(b)):
            st[k] = ("CONFLICT", "CONFLICT")
        else:
            st[k] = (bool(a), bool(b))
    cache[(run, task)] = st
    return st


out = {"meta": dict(D["meta"], eqtime=dict(T_task_s=T_TASK, source=str(SRC.relative_to(REPO)),
                                           rule="episode kept iff still a success at T_task"))}
for mode in ("full", "robot"):
    dm = D[mode]
    om = {"runs": dm["runs"]}
    for panel, task in PANEL_TASK.items():
        om[panel] = {}
        for lab, eps in dm[panel].items():
            runs = dm["runs"][panel][lab]
            assert len(runs) == 1, (mode, panel, lab, runs)
            st = status(runs[0], task)
            keep, removed, incons = {}, [], []
            for k, v in eps.items():
                s = st.get(k)
                if s is None or s[0] is not True:
                    incons.append(k); continue
                if s[1]:
                    keep[k] = v
                else:
                    removed.append(k)
            bad += len(incons)
            om[panel][lab] = keep
            log[f"{mode}/{panel}/{lab}"] = dict(run=runs[0], n_before=len(eps), n_after=len(keep),
                                                removed=removed, inconsistent=incons)
            print(f"  {mode:5s} {panel:15s} {lab:5s} {len(eps):3d} -> {len(keep):3d}"
                  f"{'  removed ' + ','.join(removed) if removed else ''}"
                  f"{'  !! INCONSISTENT ' + ','.join(incons) if incons else ''}", flush=True)
    out[mode] = om
(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "data/alpha_per_episode_tableI_eqtime.json").write_text(json.dumps(out) + "\n")
(ROOT / "scores/fig3b_filter_log.json").write_text(json.dumps(dict(inconsistent_total=bad, panels=log), indent=1) + "\n")
print(f"\ninconsistent episodes: {bad}\nwrote {ROOT / 'data/alpha_per_episode_tableI_eqtime.json'}")
sys.exit(1 if bad else 0)
