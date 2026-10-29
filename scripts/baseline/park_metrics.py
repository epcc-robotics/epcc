"""Single source of truth for PushT success and time-to-completion.

    SUCCESS  the block reached the goal pose and STAYED there for HOLD_K
             consecutive control steps, at any point in the episode.
    TTC      the sim time at which that first HOLD_K-step park COMPLETES.

Why this and not the alternatives, all of which shipped at some point:

* `first_success_step` (latched) stops the clock the first frame the block is in
  tolerance, so one brush of the goal counts. Removing the arena walls is the
  intervention that makes brush-and-lose common, so latched flatters every fast
  method. Rejected as a success criterion long ago; it survived as the TTC clock
  by accident, which left the two axes measuring different events.

* `per_env_success_held` reads the streak once at finalize(), so it means "in
  tolerance for the FINAL HOLD_K steps" — i.e. still there when the budget ran
  out. Its matching clock is the start of the LAST in-tolerance run, and that is
  what inflates TTC: an episode that parks at t=294, is disturbed, and re-parks
  at t=402 gets timed at 402 even though it had already done the task. Measured:
  it moves slippery TTC by +7 s across the board, on every method.

  It also credits leftover budget. None of these policies has a stop condition —
  action magnitude at the moment the block leaves is 0.99-1.90x its pre-park
  average for every method — so "still there at the end" partly measures how
  much budget was left to disturb it with.

* Parking, leaving and returning is not a behaviour the demonstrator exhibits
  much: 1/200 plain episodes and 20/200 slippery ones, against 43% (ours) and
  64% (SuP) of the methods' parked episodes. Timing the agent on a recovery the
  demo never had to perform is not a like-for-like comparison.

The park-completion convention (start + HOLD_K, not start) is deliberate: the
task is not done until the hold is demonstrated, and it keeps SR and TTC on the
same event. On PushT HOLD_K=10 at 0.1 s, so it adds a flat 1.0 s to every row
and changes no ordering.

CUBE IS NOT SCORED THIS WAY. `CubeGTRMetrics` defines success as grasped AND
never-dropped AND reached — a whole-episode conjunction with no hold window — so
first-reach is already the instant its own criterion certifies. `cube_ttc()`
below is unchanged behaviour, exposed here only so callers have one import.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HOLD_K = 10
POS = 20.0
ANG = np.pi / 9
DT = 0.1                      # PushT control period; cube is 0.05


def ang_err(a, g):
    r = np.abs(np.asarray(a) - np.asarray(g))
    return np.minimum(r, 2 * np.pi - r)


def park_start(m: np.ndarray, hold_k: int = HOLD_K, mode: str = "first") -> int:
    """Index at which the qualifying hold BEGINS, or -1 if there is none.

    mode="first"  the first run of >= hold_k consecutive in-tolerance frames,
                  anywhere in the episode. A run shorter than hold_k does not
                  count, which is the whole point: it separates "touched the
                  goal" from "put it there".

    mode="final"  the LAST run, and only if it reaches the final frame and is at
                  least hold_k long. This reproduces `per_env_success_held`
                  exactly ("in tolerance for the final hold_k steps") while
                  giving it a clock, so SR and TTC can be paired on that
                  criterion too. It credits a recovery: an episode that parks,
                  is disturbed, and re-parks is timed from the SECOND park.

    Both modes return the START; callers add hold_k to get the completion.
    """
    if mode == "final":
        if not m.size or not m[-1]:
            return -1
        k = len(m) - 1
        while k > 0 and m[k - 1]:
            k -= 1
        return k if (len(m) - k) >= hold_k else -1
    run = 0
    for i, v in enumerate(m):
        run = run + 1 if v else 0
        if run >= hold_k:
            return i - hold_k + 1
    return -1


def inside_from_traj(z) -> np.ndarray:
    """(T, n) in-tolerance mask from a PushT `rollout_traj.npz`."""
    blk = np.asarray(z["block_pose"], np.float64)                  # (T, n, 3)
    goal = np.asarray(z["goal_state"], np.float64).ravel()
    return ((np.linalg.norm(blk[:, :, :2] - goal[2:4], axis=2) < POS)
            & (ang_err(blk[:, :, 2], goal[4]) < ANG))


def _time_axis(run: Path, T: int, n: int, ct: float, results: dict):
    """(T+1, n) cumulative sim time, or None if it cannot be built honestly.

    SAIL and SuP vary the control period per env AND per step, so steps x a
    scalar is wrong for them; `control_ts_trace.npz` carries the real thing.

    The scalar path is only valid when the period was genuinely constant, and
    the run tells us: `control_ts_min == control_ts_max`. Returning None rather
    than falling back on a variable-rate run without a trace is the point — a
    silent scalar multiply there mis-times every env by a different amount, and
    the caller has no way to notice.
    """
    cts = run / "control_ts_trace.npz"
    if cts.is_file():
        req = np.asarray(np.load(cts)["requested"], np.float64)
        if req.shape[1] == n:
            return np.concatenate([np.zeros((1, n)), np.cumsum(req, 0)])
    lo, hi = results.get("control_ts_min"), results.get("control_ts_max")
    if lo is not None and hi is not None and abs(float(hi) - float(lo)) > 1e-9:
        return None
    return np.arange(T + 1, dtype=np.float64)[:, None] * ct * np.ones((1, n))


def park_metrics(run: Path, hold_k: int = HOLD_K, mode: str = "first",
                 ttc_at: str = "complete"):
    """Per-env park success and TTC for one PushT run directory.

    Returns None when the run is not a PushT cell or has no trajectory dump —
    callers must drop the row rather than fall back to a different definition,
    since one row timed differently is exactly the bug this module removes.
    """
    run = Path(run)
    rj, tj = run / "results.json", run / "rollout_traj.npz"
    if not (rj.is_file() and tj.is_file()):
        return None
    d = json.loads(rj.read_text())
    if "per_env_success_held" not in d:        # cube / scene use other metrics
        return None
    z = np.load(tj)
    if "block_pose" not in z or "goal_state" not in z:
        return None
    inside = inside_from_traj(z)
    T, n = inside.shape
    ct = float(d.get("control_ts_mean", DT))
    cum = _time_axis(run, T, n, ct, d)
    if cum is None:
        print(f"  [park_metrics] variable control period but no "
              f"control_ts_trace.npz — cannot time honestly: {run}")
        return None
    start = np.asarray([park_start(inside[:, e], hold_k, mode) for e in range(n)])
    ok = start >= 0
    # ttc_at="complete": the hold has to be demonstrated before the task counts
    # as done. ttc_at="start": the moment the block arrived at the pose it then
    # held — simpler to state ("when it got there and stayed"), and it drops the
    # flat hold_k * dt that is identical for every row anyway.
    off = hold_k if ttc_at == "complete" else 0
    idx = np.clip(start + off, 0, cum.shape[0] - 1)
    ttc = np.where(ok, cum[idx, np.arange(n)], np.nan)
    return dict(success=ok, ttc_s=ttc, park_start=start, inside=inside,
                episode_idx=np.asarray(z["episode_idx"], int) if "episode_idx" in z else None,
                start_step=np.asarray(z["start_step"], int) if "start_step" in z else None,
                n=n, results=d)


def demo_park_time(h5_path: str, hold_k: int = HOLD_K,
                   mode: str = "first", ttc_at: str = "complete") -> np.ndarray:
    """(n_episodes,) FRAME at which each demo's first hold_k park completes.

    -1 where the demo never parks. Multiply by DT for seconds, and subtract the
    eval's `start_step` before doing so — the eval resets mid-episode, so the
    demonstrator's REMAINING time from that state is the comparable quantity,
    not its whole-episode duration.
    """
    import hdf5plugin  # noqa: F401
    import h5py
    with h5py.File(h5_path, "r") as f:
        off, ln = f["ep_offset"][:].astype(int), f["ep_len"][:].astype(int)
        blk = f["block_pose"][:].astype(np.float64)
        goal = f["goal_pose"][:].astype(np.float64)
    ok = ((np.linalg.norm(blk[:, :2] - goal[:, :2], axis=1) < POS)
          & (ang_err(blk[:, 2], goal[:, 2]) < ANG))
    out = np.full(len(ln), -1.0)
    for i in range(len(ln)):
        s = park_start(ok[off[i]:off[i] + ln[i]], hold_k, mode)
        out[i] = (s + (hold_k if ttc_at == "complete" else 0)) if s >= 0 else -1
    return out


def demo_line_s(runs, demo_frames: np.ndarray) -> float:
    """Demonstrator TTC for a panel, in seconds, METHOD-AGNOSTICALLY.

    Every distinct (episode, start_step) any row was evaluated on counts once,
    whether or not any method solved it. Keying on the pair and not the episode
    matters: SAIL repeats use seeds 42/43/44 and draw different inits, so the
    union across ten cells is 115 distinct states, not 50.
    """
    seen: dict[tuple[int, int], float] = {}
    for run in runs:
        tj = Path(run) / "rollout_traj.npz"
        if not tj.is_file():
            continue
        z = np.load(tj)
        if "episode_idx" not in z:
            continue
        ep, st = np.asarray(z["episode_idx"], int), np.asarray(z["start_step"], int)
        dd = demo_frames[ep] - st
        for e in np.flatnonzero((demo_frames[ep] >= 0) & (dd > 0)):
            seen[(int(ep[e]), int(st[e]))] = float(dd[e] * DT)
    return float(np.median(list(seen.values()))) if seen else np.nan


def alpha_paired(run, demo_frames: np.ndarray, hold_k: int = HOLD_K,
                 mode: str = "first"):
    """Per-episode speedup samples demo/ours for one run, both on the park clock.

    Paired, not a ratio of medians: env i is NOT demo i (the eval draws episodes
    through its own RNG), so the pairing goes through `episode_idx`.
    """
    m = park_metrics(run, hold_k, mode)
    if m is None or m["episode_idx"] is None:
        return np.asarray([])
    ep, st = m["episode_idx"], m["start_step"]
    dd = (demo_frames[ep] - st) * DT
    sel = m["success"] & (demo_frames[ep] >= 0) & (dd > 0) & np.isfinite(m["ttc_s"]) \
        & (m["ttc_s"] > 0)
    return dd[sel] / m["ttc_s"][sel]


def cube_ttc(run: Path):
    """Cube: success_full + first-reach TTC. Unchanged — see the module docstring."""
    run = Path(run)
    d = json.loads((run / "results.json").read_text())
    ok = np.asarray(d.get("success_full", d.get("per_env_success")), bool)
    per_env = d.get("ttc_sim_time_actual_per_env_s")
    if per_env is not None and len(per_env) == len(ok):
        ttc = np.asarray(per_env, float)
    else:
        ct = float(d.get("control_ts_mean", 0.05))
        ttc = (np.asarray(d["first_success_step"], float) + 1.0) * ct
    return dict(success=ok, ttc_s=ttc, results=d)
