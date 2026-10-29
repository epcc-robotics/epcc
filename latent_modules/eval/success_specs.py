"""Per-task success-criterion registry.

A SuccessSpec is a named callable that decides whether one env's running
behavioral state satisfies one notion of success.

Each task can register multiple specs (e.g., cube has ``naive`` from
``world.terminateds`` and ``gtr`` = grasp + transport + reach). The plotting
script picks one as the headline metric.

State convention (passed to ``check_step`` each env-step):
    {
        "t":         int,                  # current step
        "cube_pos":  np.ndarray (3,),      # task-specific keys ok
        "effector_pos": np.ndarray (3,),
        "gripper_opening": float,
        "terminated":  bool,               # raw env terminated signal
        ...
    }

A spec is satisfied for the **episode** once ``check_episode`` (which sees the
full running list of step-dicts) returns True. ``first_step`` is the smallest
t at which the criterion first became True and remained True until end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Base spec
# ---------------------------------------------------------------------------

StepFn = Callable[[List[dict], dict], bool]
"""Callable signature: (states_so_far, latest_state) -> bool.

``states_so_far`` is the running list of per-step state dicts for one env;
``latest_state`` is the dict just appended. The function returns True if the
criterion is satisfied AT THIS STEP given the running history.
"""


@dataclass
class SuccessSpec:
    name: str
    """Short identifier (e.g., ``"naive"``, ``"gtr"``). Written to npz keys."""

    check_step: StepFn
    """Per-step check. Used to compute first_success_step + per-step trace."""

    description: str = ""
    """Free-text description for meta.yaml provenance."""


# ---------------------------------------------------------------------------
# Cube AW / MV: naive + GTR
# ---------------------------------------------------------------------------

def _cube_naive(states: List[dict], latest: dict) -> bool:
    """OGBench cube's built-in terminated signal — fires on cube-in-goal-pose
    without checking gripper release or arm retreat. Slip-release false
    positives possible."""
    return bool(latest.get("terminated", False))


def _cube_gtr(
    states: List[dict],
    latest: dict,
    *,
    lift_thresh: float = 0.02,
    min_lift_steps: int = 3,
    hold_dist_thresh: float = 0.05,
    drop_jump_thresh: float = 0.02,
    goal_dist_thresh: float = 0.07,
) -> bool:
    """Strict 3-criterion success: Grasped & no-drop Transport & Reached goal.

    Mirrors ``eval_mpc_multiview._compute_episode_metrics`` evaluated up to
    the latest step. Same thresholds as the canonical scorer. Returns True
    only when all three sub-criteria are met for the running history.
    """
    goal_pos = latest.get("goal_pos")
    table_z = latest.get("table_z")
    if goal_pos is None or table_z is None:
        return False
    if not states:
        return False

    cube_pos = np.array([s["cube_pos"] for s in states])
    cube_z = cube_pos[:, 2]
    eff_pos = np.array([s["effector_pos"] for s in states])
    eff_cube_dist = np.linalg.norm(eff_pos - cube_pos, axis=1)

    lifted = cube_z > (table_z + lift_thresh)
    near = eff_cube_dist < hold_dist_thresh
    grasp = lifted & near
    # Longest grasp run length.
    longest = 0
    cur = 0
    for v in grasp.tolist():
        if v:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    grasped = longest >= min_lift_steps
    if not grasped:
        return False

    first_grasp = int(np.argmax(grasp)) if grasp.any() else -1
    no_drop = True
    if first_grasp >= 0:
        ddist = np.abs(np.diff(eff_cube_dist))
        elevation_guard = table_z + 0.03
        placement_proximity = 0.12
        goal_arr = np.asarray(goal_pos)
        for i in range(first_grasp, len(states) - 1):
            cube_goal_dist = np.linalg.norm(cube_pos[i] - goal_arr)
            if cube_goal_dist < placement_proximity:
                continue
            if ddist[i] > drop_jump_thresh and cube_z[i] > elevation_guard:
                no_drop = False
                break
    if not no_drop:
        return False

    final_dist = float(np.linalg.norm(cube_pos[-1] - np.asarray(goal_pos)))
    reached_goal = final_dist < goal_dist_thresh
    return bool(grasped and no_drop and reached_goal)


# ---------------------------------------------------------------------------
# PushT / generic: terminated-only fallback
# ---------------------------------------------------------------------------

def _terminated_only(states: List[dict], latest: dict) -> bool:
    return bool(latest.get("terminated", False))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SUCCESS_SPECS: Dict[str, List[SuccessSpec]] = {
    "cube": [
        SuccessSpec(name="naive", check_step=_cube_naive,
                    description="OGBench terminated signal (single-condition)."),
        SuccessSpec(name="gtr",   check_step=_cube_gtr,
                    description="Grasp + no-drop Transport + Reached goal."),
    ],
    "cube-mv": [
        SuccessSpec(name="naive", check_step=_cube_naive,
                    description="OGBench terminated signal (single-condition)."),
        SuccessSpec(name="gtr",   check_step=_cube_gtr,
                    description="Grasp + no-drop Transport + Reached goal."),
    ],
    "pusht": [
        SuccessSpec(name="naive", check_step=_terminated_only,
                    description="Env terminated (IoU >= threshold)."),
    ],
    # Stubs — tasks not yet wired up. The plotting script will treat them as
    # naive-only until proper specs are registered.
    "scene-seq": [
        SuccessSpec(name="naive", check_step=_terminated_only,
                    description="Env terminated."),
    ],
    "rope": [
        SuccessSpec(name="naive", check_step=_terminated_only,
                    description="Env terminated."),
    ],
    "granular": [
        SuccessSpec(name="naive", check_step=_terminated_only,
                    description="Env terminated."),
    ],
    "pusht-lowfric": [
        SuccessSpec(name="naive", check_step=_terminated_only,
                    description="Env terminated."),
    ],
}


def get_success_spec(task: str) -> List[SuccessSpec]:
    """Look up the list of registered success specs for a task.

    Falls back to a single ``naive`` (terminated-only) spec if unregistered,
    so a new task works out-of-the-box and we add a stricter spec later.
    """
    if task in SUCCESS_SPECS:
        return SUCCESS_SPECS[task]
    return [SuccessSpec(name="naive", check_step=_terminated_only,
                        description=f"(auto) Env terminated — no spec registered for task={task!r}.")]
