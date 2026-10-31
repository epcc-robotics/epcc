"""Equal-time scoring shared by every script in findings/261025_equal_time/ (2026-10-25).

Every method in a task gets the SAME simulated time T_TASK (seconds), whatever its control
period. T_TASK is the shortest horizon any Table I row was run for, so it is reached by
truncating saved rollouts; no rollout is re-run. That is exact because no policy reads the
budget (only the eval loop and the metrics do), so a shorter run is a prefix of the saved one.

  PushT    43.5 s  (EPCC / Play+Goal ran 435 steps; DP / SkiP / SuP ran 489 = 48.9 s)
  Slippery 59.4 s  (594 steps for every method; SAIL's precision slowdown ran 59.4-97.3 s)
  Cube     15.7 s  (314 steps for every method; SAIL ran 15.9-18.5 s)

Clock = the eval's own `requested` per-step period (control_ts_trace.npz), as in park_metrics
and the eval's ttc_sim_time_actual_per_env_s. Constant-period runs without a trace use
steps x period.

PushT family: park_metrics(hold_k, mode='first', ttc_at='complete') with the Table I hold rule
hold_k = round(5.0 / control_ts_mean) (50 at 0.1 s; SAIL's time-matched value otherwise).
Truncating at T keeps a success iff its park completes by T, i.e. success_T = ok & (ttc <= T):
the first qualifying park is unchanged by truncation if it completes inside the window, and no
park can complete inside the window if the first one does not.

Cube: CubeGTRMetrics (policy/_metrics.py) is a whole-episode conjunction judged at the LAST
state, so it is recomputed from rollout_traj.npz on the states up to T. full_length=True must
reproduce the eval's own success_full and first_success_step (checked by the callers).

TP = (sum over successes of 1/TTC - n_fail / t_max) / n  (SAIL Eq. 2), t_max = T_TASK.
"""
import json
import os
from itertools import groupby
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
import sys  # noqa: E402
sys.path.insert(0, str(REPO / "scripts/baseline"))
import park_metrics as pm  # noqa: E402

T_TASK = {"pusht": 43.5, "slippery": 59.4, "cube": 15.7}
EPS = 1e-6
TARGET_S = 5.0
HERE = Path(__file__).resolve().parent
# Published run directories: a tree with the research layout findings/<campaign>/..., i.e. the
# Hugging Face release's published_runs/ (EPCC_PUBLISHED). Outputs go to EPCC_PAPER_OUT/table1_equal_time.
PUB = Path(os.environ.get("EPCC_PUBLISHED", REPO))
ROOT = Path(os.environ.get("EPCC_PAPER_OUT", REPO / "results" / "paper")) / "table1_equal_time"


def push_metrics(run):
    """Per-env PushT-family scoring at the run's own length (Table I convention)."""
    run = Path(run)
    m1 = pm.park_metrics(run, hold_k=1)
    if m1 is None:
        raise RuntimeError(f"park_metrics returned None: {run}")
    ct = float(m1["results"].get("control_ts_mean", pm.DT))
    hk = max(1, int(round(TARGET_S / ct)))
    m = pm.park_metrics(run, hold_k=hk, mode="first", ttc_at="complete")
    steps = int(np.load(run / "rollout_traj.npz")["action"].shape[0])
    return dict(ok=np.asarray(m["success"], bool), ttc=np.asarray(m["ttc_s"], float),
                latch_ok=np.asarray(m1["success"], bool), latch_ttc=np.asarray(m1["ttc_s"], float),
                ep=np.asarray(m["episode_idx"], int), st=np.asarray(m["start_step"], int),
                hk=hk, ct=ct, steps=steps, tmax_own=steps * ct, n=int(m["n"]))


def cut(ok, ttc, T):
    """Success at simulated time T: the park (or first reach, Cube) completes by T."""
    return np.asarray(ok, bool) & np.isfinite(ttc) & (np.asarray(ttc, float) <= T + EPS)


def _cube_time_axis(run, steps, n):
    tr = Path(run) / "control_ts_trace.npz"
    if tr.is_file():
        req = np.asarray(np.load(tr)["requested"], np.float64)
        assert req.shape == (steps, n), (run, req.shape, steps, n)
        return np.concatenate([np.zeros((1, n)), np.cumsum(req, 0)])
    d = json.loads((Path(run) / "results.json").read_text())
    lo, hi = d.get("control_ts_min", 0.05), d.get("control_ts_max", 0.05)
    assert abs(float(hi) - float(lo)) < 1e-9, f"variable period but no trace: {run}"
    return np.arange(steps + 1, dtype=np.float64)[:, None] * float(lo) * np.ones((1, n))


# CubeGTRMetrics thresholds, copied from policy/_metrics.py (class constants)
LIFT, MIN_LIFT, HOLD_D, DROP_J, GOAL_D = 0.02, 3, 0.05, 0.02, 0.07


def _gtr_one(cube, eff, goal, table_z):
    """CubeGTRMetrics._compute_one + _first_reached_step, line for line, on (L,3) arrays."""
    if len(cube) == 0:
        return False, -1
    cube_z = cube[:, 2]
    d_ec = np.linalg.norm(eff - cube, axis=1)
    grasp = (cube_z > table_z + LIFT) & (d_ec < HOLD_D)
    longest = max((sum(1 for _ in g) for k, g in groupby(grasp.tolist()) if k), default=0)
    grasped = longest >= MIN_LIFT
    first_grasp = int(np.argmax(grasp)) if grasp.any() else -1
    no_drop = False
    if grasped and first_grasp >= 0:
        no_drop = True
        ddist = np.abs(np.diff(d_ec))
        for i in range(first_grasp, len(cube) - 1):
            if float(np.linalg.norm(cube[i] - goal)) < 0.12:
                continue
            if ddist[i] > DROP_J and cube_z[i] > table_z + 0.03:
                no_drop = False
                break
    reached = float(np.linalg.norm(cube[-1] - goal)) < GOAL_D
    ok = bool(grasped and no_drop and reached)
    fr = -1
    if ok:
        dd = np.linalg.norm(cube - goal, axis=1)
        hit = np.flatnonzero(dd < GOAL_D)
        fr = int(hit[0]) if hit.size else -1
    return ok, fr


def cube_metrics(run, T=None):
    """Per-env Cube GTR success and first-reach TTC, on the states up to sim time T (None = all)."""
    run = Path(run)
    z = np.load(run / "rollout_traj.npz")
    cube = np.asarray(z["cube_pos"], np.float64); eff = np.asarray(z["effector_pos"], np.float64)
    goal = np.asarray(z["goal_pos"], np.float64); tz = np.asarray(z["table_z"], np.float64)
    steps, n = cube.shape[0], cube.shape[1]
    cum = _cube_time_axis(run, steps, n)
    ok = np.zeros(n, bool); ttc = np.full(n, np.nan); keep = np.zeros(n, int)
    for e in range(n):
        L = steps if T is None else int(np.sum(cum[1:, e] <= T + EPS))   # state i is at cum[i+1]
        keep[e] = L
        o, fr = _gtr_one(cube[:L, e], eff[:L, e], goal[e], float(tz[e]))
        ok[e] = o
        if o and fr >= 0:
            ttc[e] = cum[fr + 1, e]
    ct_hdr = 0.05
    return dict(ok=ok, ttc=ttc, ep=np.asarray(z["episode_idx"], int), st=np.asarray(z["start_step"], int),
                steps=steps, n=n, kept_states=keep, tmax_own=steps * ct_hdr,
                sim_time_end=cum[-1].copy())


def cube_reference(run):
    """What the paper used: the eval's own success_full + cube_ttc TTC."""
    m = pm.cube_ttc(Path(run))
    return np.asarray(m["success"], bool), np.asarray(m["ttc_s"], float)


def summarize(ok, ttc, tmax, min_ok=None):
    ok = np.asarray(ok, bool); ttc = np.asarray(ttc, float)
    n, s = int(ok.size), int(ok.sum())
    t = ttc[ok]
    tp = (sum(1.0 / max(x, 1e-9) for x in t) - (n - s) / tmax) / n
    med = float(np.median(t)) if t.size and (min_ok is None or t.size >= min_ok) else float("nan")
    return dict(SR_pct=100.0 * s / n, TTC_median_s=med, TP=float(tp), n=n, n_success=s, t_max_s=float(tmax))
