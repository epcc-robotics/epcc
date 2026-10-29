"""Task-level metrics collectors for the unified v2 in-process eval loop.

Single file containing both the base implementations (moved from the retired
``policy._metrics`` module) AND the Phase-3 extensions that add TTC +
control_timestep tracking.

Public classes:
  - ``TaskMetrics``           — abstract base for all metric collectors
  - ``PushTCoverageMetrics``  — PushT coverage + TTC
  - ``CubeGTRMetrics``        — Cube Grasp-Transport-Reach + TTC
"""
from __future__ import annotations

import abc
from itertools import groupby
from typing import Any, List, Optional

import numpy as np

__all__ = ["TaskMetrics", "PushTCoverageMetrics", "CubeGTRMetrics", "GranularChamferMetrics", "SceneMetrics", "SceneStageMetrics"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _per_env(val: Any, env_idx: int, n_envs: int) -> Any:
    """Index a value that may be a scalar or a per-env array of length n_envs."""
    if np.ndim(val) == 0:
        return val
    arr = np.asarray(val)
    if arr.shape and arr.shape[0] == n_envs:
        return arr[env_idx]
    return val


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TaskMetrics(abc.ABC):
    @abc.abstractmethod
    def update(
        self,
        obs: dict,
        info: dict,
        action: np.ndarray,
        step: int,
        env_mask: np.ndarray,
    ) -> None:
        """Called once per env-step for all active envs."""

    @abc.abstractmethod
    def finalize(self) -> dict:
        """Return the aggregated metrics dict after all episodes complete."""


# ---------------------------------------------------------------------------
# Scene (OGBench) — terminated-only provisional success
# ---------------------------------------------------------------------------

class SceneMetrics(TaskMetrics):
    """Terminated-only success collector for OGBench Scene.

    Scene success ("all objects in target config") surfaces as the env's native
    ``terminated`` flag. The stricter task-specific criterion is deferred pending
    visual inspection of rollouts, so this reports the terminated signal
    (success > terminated > done) as a provisional SR + TTC. Mirrors the
    terminated-only path used by ``PushTCoverageMetrics`` without coverage.

    finalize() output keys
    ----------------------
    success_rate         : float
    per_env_success      : (n_envs,) bool
    first_success_step   : (n_envs,) int  — first terminated step, else -1
    ttc_steps_mean       : float (mean over successful envs)
    ttc_sim_time_mean_s  : float
    control_timestep_s   : float
    n_envs               : int
    """

    def __init__(self, n_envs: int, control_timestep_s: float = 0.05) -> None:
        self.n_envs = n_envs
        self._control_timestep_s = float(control_timestep_s)
        self._success: np.ndarray = np.zeros(n_envs, dtype=bool)
        self._first_success_step: np.ndarray = np.full(n_envs, -1, dtype=np.int64)

    def _extract_success(self, info: dict, env_idx: int) -> bool:
        """Read per-env terminated flag. Prefers the eval loop's private
        ``_scene_terminated`` (clean (n_envs,) bool); falls back to native
        env keys, coercing a possible history dim to the latest frame."""
        for key in ("_scene_terminated", "success", "terminated", "done"):
            if key in info:
                v = np.asarray(_per_env(info[key], env_idx, self.n_envs))
                return bool(v.ravel()[-1]) if v.size else False
        return False

    def update(
        self,
        obs: dict,
        info: dict,
        action: np.ndarray,
        step: int,
        env_mask: np.ndarray,
    ) -> None:
        for env_idx in range(self.n_envs):
            if not env_mask[env_idx]:
                continue
            if not self._success[env_idx] and self._extract_success(info, env_idx):
                self._success[env_idx] = True
                self._first_success_step[env_idx] = step

    def finalize(self) -> dict:
        out = {
            "success_rate": float(self._success.mean()),
            "per_env_success": self._success.copy(),
            "first_success_step": self._first_success_step.copy(),
            "n_envs": self.n_envs,
        }
        success = out["per_env_success"].astype(bool)
        if success.any():
            steps = out["first_success_step"][success]
            ttc_steps = float(np.mean(steps))
            out["ttc_steps_mean"] = ttc_steps
            out["ttc_sim_time_mean_s"] = ttc_steps * self._control_timestep_s
        else:
            out["ttc_steps_mean"] = float("nan")
            out["ttc_sim_time_mean_s"] = float("nan")
        out["control_timestep_s"] = self._control_timestep_s
        return out


# ---------------------------------------------------------------------------
# Scene (OGBench) — per-stage physical success
# ---------------------------------------------------------------------------

class SceneStageMetrics(TaskMetrics):
    """Per-stage physical success for the OGBench Scene 5-subtask sequence.

    Replaces ``SceneMetrics``' terminated-only signal, which is *invalid* for
    this task: the env's ``terminated`` compares against the randomised targets
    drawn at ``reset`` (``_target_drawer_pos`` / ``_target_window_pos`` /
    ``_target_button_states`` / cube target mocap), and the scene eval callables
    never set those. Ground-truth expert replays therefore score 0 too, so
    ``terminated`` measures nothing about controller quality.

    This collector instead evaluates the *fixed physical* predicates that the
    demo collector itself used (``check_physical_success`` in
    ``collect_multiview_diffusion_scene.py``) directly against live env state,
    stage by stage:

        A open_drawer       drawer_slide qpos < -0.12
        B cube_into_drawer  _is_in_drawer(block_0_pos)   (drawer-relative box)
        C close_drawer      |drawer_slide qpos| <= 0.04
        D press_window_btn  latched button 1 == 1
        E open_window       |window_slide qpos - 0.2| <= 0.04

    Stages are *chained*: stage k only counts once every earlier in-segment
    stage has already been observed. Without chaining the labels are
    meaningless — e.g. the drawer is trivially "closed" at t=0, long before
    stage C is actually performed.

    ``segment`` selects which contiguous run of stages this rollout is scored
    on (``"ABCDE"`` = full task, ``"BC"`` = cube-into-drawer then close, ...).
    Stages before the segment are assumed already satisfied by the segment's
    initial state and are not scored.

    update() reads ``info["behavioral"]`` which must contain:
        drawer_qpos      : float    drawer_slide joint qpos
        window_qpos      : float    window_slide joint qpos
        block_pos        : (3,)     object_joint_0 qpos[:3]
        drawer_handle_y  : float    site_xpos['drawer_handle_center'][1]
        button1_state    : int      latched _cur_button_states[1]
        terminated       : bool     (recorded, not used for success)

    finalize() output keys
    ----------------------
    success_rate            : float  — segment success (final in-segment stage reached)
    per_env_success         : (n_envs,) bool
    success_rate_held       : float  — segment success AND still holding at the last step
    stage_reached_rate      : dict stage -> float
    stage_reached           : dict stage -> (n_envs,) bool
    stage_first_step        : dict stage -> (n_envs,) int (-1 if never)
    stage_true_at_start     : dict stage -> (n_envs,) bool  — degenerate-label guard
    first_success_step      : (n_envs,) int
    ttc_steps_mean          : float (mean over successful envs)
    ttc_sim_time_mean_s     : float
    full_physical_success   : (n_envs,) bool — all 5 collector conditions at once
    segment                 : str
    control_timestep_s      : float
    n_envs                  : int
    """

    STAGES = ("A", "B", "C", "D", "E")

    # Verbatim from the env / collector — keep in sync with
    # scripts/baseline/label_scene_stages.py.
    DRAWER_OPEN_QPOS: float = -0.12
    TOL: float = 0.04
    WINDOW_OPEN_QPOS: float = 0.2
    DRAWER_BOX_LOW_XZ = (0.21, 0.0)
    DRAWER_BOX_HIGH_XZ = (0.45, 0.15)
    DRAWER_BOX_DY = (-0.27, -0.07)

    def __init__(
        self,
        n_envs: int,
        control_timestep_s: float = 0.05,
        segment: str = "ABCDE",
        hold_window: int = 20,
    ) -> None:
        seg = str(segment).upper().strip()
        if not seg or any(c not in self.STAGES for c in seg):
            raise ValueError(f"segment={segment!r} must be drawn from {self.STAGES}")
        idx = [self.STAGES.index(s) for s in seg]
        if idx != list(range(idx[0], idx[0] + len(idx))):
            raise ValueError(
                f"segment={segment!r} must be a contiguous ascending run of stages"
            )
        self.n_envs = n_envs
        self.segment = seg
        self._seg_idx = idx
        self._control_timestep_s = float(control_timestep_s)

        n_seg = len(idx)
        self._reached = np.zeros((n_envs, n_seg), dtype=bool)
        self._first_step = np.full((n_envs, n_seg), -1, dtype=np.int64)
        self._true_at_start = np.zeros((n_envs, n_seg), dtype=bool)
        self._seen = np.zeros(n_envs, dtype=bool)   # first update seen per env
        self._last_hold = np.zeros((n_envs, n_seg), dtype=bool)
        self._full_success = np.zeros(n_envs, dtype=bool)
        # Collateral damage. A segment run inherits the earlier stages as its
        # initial condition; undoing one of them is not a neutral failure, it is
        # a worse outcome than doing nothing. Two forms are tracked:
        #   wrong_button — button 0 (the drawer lock) is latched to 1 at
        #     collection reset and the expert never touches it. Pressing it sets
        #     drawer damping to 1e6, i.e. locks the drawer shut. A controller
        #     that mashes buttons could satisfy stage D by accident while
        #     destroying stages A/C, and a button1-only predicate would score
        #     that as success.
        #   prior_broken — a stage before the segment that held at t=0 stops
        #     holding (cube knocked out of the drawer, drawer dragged open, ...).
        self._wrong_button = np.zeros(n_envs, dtype=bool)
        self._prior_at_start = np.zeros((n_envs, 5), dtype=bool)
        self._prior_broken = np.zeros(n_envs, dtype=bool)
        # "Still holding on the LAST step" is the wrong criterion for a segment:
        # the segment's reference ends at its goal, so once the goal is reached
        # the tracker has nothing left to follow and the arm loiters on top of
        # what it just achieved. Measured on segment A, that loitering pushed the
        # drawer shut again in 12 of 14 openings — while the SAME controller on
        # the full task kept 3 of 4 open, because there the reference moved on to
        # the next subtask and carried the arm away. So `held_at_end` was scoring
        # an artefact of where the segment was cut.
        # `held_window` instead asks whether the goal survived the next
        # `hold_window` steps after it was first reached — long enough to rule out
        # a momentary brush, short enough not to punish having nothing left to do.
        self._hold_window = int(hold_window)
        self._final_since = np.full(n_envs, -1, dtype=np.int64)
        self._final_hold_ok = np.zeros(n_envs, dtype=bool)
        self._final_hold_seen = np.zeros(n_envs, dtype=np.int64)

    # -- predicates ---------------------------------------------------------

    def _is_in_drawer(self, block_pos: np.ndarray, handle_y: float) -> bool:
        lo_x, lo_z = self.DRAWER_BOX_LOW_XZ
        hi_x, hi_z = self.DRAWER_BOX_HIGH_XZ
        dy_lo, dy_hi = self.DRAWER_BOX_DY
        x, y, z = float(block_pos[0]), float(block_pos[1]), float(block_pos[2])
        return (
            lo_x <= x <= hi_x
            and (handle_y + dy_lo) <= y <= (handle_y + dy_hi)
            and lo_z <= z <= hi_z
        )

    def _conditions(self, beh: dict, env_idx: int) -> np.ndarray:
        """(5,) bool — the five stage predicates evaluated on the live state."""
        drawer = float(_per_env(beh["drawer_qpos"], env_idx, self.n_envs))
        window = float(_per_env(beh["window_qpos"], env_idx, self.n_envs))
        block = np.asarray(
            _per_env(beh["block_pos"], env_idx, self.n_envs), dtype=np.float64
        )
        handle_y = float(_per_env(beh["drawer_handle_y"], env_idx, self.n_envs))
        button1 = int(_per_env(beh["button1_state"], env_idx, self.n_envs))
        return np.array([
            drawer < self.DRAWER_OPEN_QPOS,
            self._is_in_drawer(block, handle_y),
            abs(drawer) <= self.TOL,
            button1 == 1,
            abs(window - self.WINDOW_OPEN_QPOS) <= self.TOL,
        ], dtype=bool)

    # -- interface ----------------------------------------------------------

    def update(
        self,
        obs: dict,
        info: dict,
        action: np.ndarray,
        step: int,
        env_mask: np.ndarray,
    ) -> None:
        if "behavioral" not in info:
            raise RuntimeError(
                "SceneStageMetrics.update() requires info['behavioral']. "
                "Got keys: " + str(list(info.keys()))
            )
        beh = info["behavioral"]
        for env_idx in range(self.n_envs):
            if not env_mask[env_idx]:
                continue
            cond = self._conditions(beh, env_idx)

            # Collector's own criterion: all five simultaneously. button 0 is
            # latched to 1 at collection reset and never toggled, so it is a
            # free condition and is not re-checked here.
            if cond.all():
                self._full_success[env_idx] = True

            # Wrong-button press: button 0 must stay latched at 1 all the way
            # through. Any observation of 0 means the drawer got locked.
            if int(_per_env(beh.get("button0_state", 1), env_idx, self.n_envs)) != 1:
                self._wrong_button[env_idx] = True

            seg_cond = cond[self._seg_idx]
            self._last_hold[env_idx] = seg_cond
            if not self._seen[env_idx]:
                self._seen[env_idx] = True
                self._true_at_start[env_idx] = seg_cond
                self._prior_at_start[env_idx] = cond
            # Stages BEFORE the segment are its inherited initial condition;
            # losing one of them invalidates the run even if the segment's own
            # predicate later fires.
            first = self._seg_idx[0]
            if first > 0:
                inherited = self._prior_at_start[env_idx, :first]
                if np.any(inherited & ~cond[:first]):
                    self._prior_broken[env_idx] = True

            # Post-goal hold window: once the segment's last stage has latched,
            # every one of the next `hold_window` steps must keep holding it.
            if (
                self._final_since[env_idx] >= 0
                and self._final_hold_seen[env_idx] < self._hold_window
            ):
                self._final_hold_seen[env_idx] += 1
                if not seg_cond[-1]:
                    self._final_hold_ok[env_idx] = False

            # Chained: stage k can only latch once k-1 has already latched.
            for k in range(len(self._seg_idx)):
                if self._reached[env_idx, k]:
                    continue
                if k > 0 and not self._reached[env_idx, k - 1]:
                    break
                if seg_cond[k]:
                    self._reached[env_idx, k] = True
                    self._first_step[env_idx, k] = step
                    if k == len(self._seg_idx) - 1:
                        # Segment goal just reached — arm the hold window.
                        self._final_since[env_idx] = step
                        self._final_hold_ok[env_idx] = True
                        self._final_hold_seen[env_idx] = 0
                else:
                    break

    def finalize(self) -> dict:
        n_seg = len(self._seg_idx)
        success = self._reached[:, -1].copy()
        first_success_step = self._first_step[:, -1].copy()
        held = success & self._last_hold[:, -1]
        # `held_window` is the defensible hold criterion for a segment; see the
        # note in __init__ on why `held_at_end` is not.
        held_window = success & self._final_hold_ok
        # The honest headline: reached the segment goal, kept it through the
        # hold window, and did not destroy anything it was handed.
        clean = held_window & ~self._wrong_button & ~self._prior_broken

        out: dict = {
            "success_rate": float(success.mean()),
            "per_env_success": success,
            "success_rate_held": float(held.mean()),
            "per_env_success_held": held,
            "success_rate_held_window": float(held_window.mean()),
            "per_env_success_held_window": held_window,
            "hold_window": self._hold_window,
            "success_rate_clean": float(clean.mean()),
            "per_env_success_clean": clean,
            "wrong_button_rate": float(self._wrong_button.mean()),
            "per_env_wrong_button": self._wrong_button.copy(),
            "prior_stage_broken_rate": float(self._prior_broken.mean()),
            "per_env_prior_stage_broken": self._prior_broken.copy(),
            "first_success_step": first_success_step,
            "full_physical_success": self._full_success.copy(),
            "full_physical_success_rate": float(self._full_success.mean()),
            "segment": self.segment,
            "n_envs": self.n_envs,
            "control_timestep_s": self._control_timestep_s,
        }

        stage_reached, stage_rate, stage_first, stage_start = {}, {}, {}, {}
        for k in range(n_seg):
            s = self.STAGES[self._seg_idx[k]]
            stage_reached[s] = self._reached[:, k].copy()
            stage_rate[s] = float(self._reached[:, k].mean())
            stage_first[s] = self._first_step[:, k].copy()
            stage_start[s] = self._true_at_start[:, k].copy()
        out["stage_reached"] = stage_reached
        out["stage_reached_rate"] = stage_rate
        out["stage_first_step"] = stage_first
        out["stage_true_at_start"] = stage_start

        if success.any():
            ttc_steps = float(np.mean(first_success_step[success]))
            out["ttc_steps_mean"] = ttc_steps
            out["ttc_sim_time_mean_s"] = ttc_steps * self._control_timestep_s
        else:
            out["ttc_steps_mean"] = float("nan")
            out["ttc_sim_time_mean_s"] = float("nan")
        return out


# ---------------------------------------------------------------------------
# PushT coverage base + Phase-3 extension (combined)
# ---------------------------------------------------------------------------

class PushTCoverageMetrics(TaskMetrics):
    """Per-env coverage tracker with TTC + DP-paper canonical aggregates.

    finalize() output keys
    ----------------------
    success_rate         : float
    per_env_success      : (n_envs,) bool
    first_success_step   : (n_envs,) int   — timeout_penalty_step on failure
    max_score            : (n_envs,) float — max coverage over trajectory
    final_score          : (n_envs,) float — coverage at last step recorded
    distances            : (n_envs, eval_budget) float
    n_envs               : int
    ttc_steps_mean       : float (mean over successful envs)
    ttc_sim_time_mean_s  : float (= ttc_steps_mean * control_timestep_s)
    control_timestep_s   : float
    mean_max_coverage    : float (DP-paper test/mean_score)
    mean_final_coverage  : float
    """

    def __init__(
        self,
        n_envs: int,
        eval_budget: int,
        control_timestep_s: float = 0.1,
        timeout_penalty_step: Optional[int] = None,
    ) -> None:
        self.n_envs = n_envs
        self.eval_budget = eval_budget
        self._control_timestep_s = float(control_timestep_s)
        self.timeout_penalty_step: int = (
            timeout_penalty_step if timeout_penalty_step is not None else eval_budget
        )
        self._success: np.ndarray = np.zeros(n_envs, dtype=bool)
        self._first_success_step: np.ndarray = np.full(
            n_envs, self.timeout_penalty_step, dtype=np.int64
        )
        self._max_score: np.ndarray = np.zeros(n_envs, dtype=np.float64)
        self._final_score: np.ndarray = np.zeros(n_envs, dtype=np.float64)
        self._distances: np.ndarray = np.zeros((n_envs, eval_budget), dtype=np.float64)
        # final-frame-stabilized success (un-latched): last-step in-tol flag +
        # consecutive-in-tol streak for the held@K variant. Complements the
        # latched _success (ever-reached) so drift-after-reach is caught.
        self._last_success: np.ndarray = np.zeros(n_envs, dtype=bool)
        self._held_streak: np.ndarray = np.zeros(n_envs, dtype=np.int64)
        self._held_k: int = 10

    def _extract_coverage(self, info: dict, env_idx: int) -> float:
        """Read per-env coverage from info, with legacy reward fallback."""
        for key in ("coverage", "reward"):
            if key in info:
                return float(_per_env(info[key], env_idx, self.n_envs))
        return 0.0

    def _extract_success(self, info: dict, env_idx: int) -> bool:
        """Read per-env success/terminated flag (success > terminated > done)."""
        for key in ("success", "terminated", "done"):
            if key in info:
                return bool(_per_env(info[key], env_idx, self.n_envs))
        return False

    def update(
        self,
        obs: dict,
        info: dict,
        action: np.ndarray,
        step: int,
        env_mask: np.ndarray,
    ) -> None:
        col = min(step, self.eval_budget - 1)
        for env_idx in range(self.n_envs):
            if not env_mask[env_idx]:
                continue
            cov = self._extract_coverage(info, env_idx)
            self._distances[env_idx, col] = cov
            if cov > self._max_score[env_idx]:
                self._max_score[env_idx] = cov
            self._final_score[env_idx] = cov
            cur_success = self._extract_success(info, env_idx)
            # un-latched: current-step flag + consecutive-in-tol streak
            self._last_success[env_idx] = cur_success
            self._held_streak[env_idx] = (
                self._held_streak[env_idx] + 1 if cur_success else 0
            )
            if not self._success[env_idx] and cur_success:
                self._success[env_idx] = True
                self._first_success_step[env_idx] = step

    def finalize(self) -> dict:
        out = {
            "success_rate": float(self._success.mean()),
            "per_env_success": self._success.copy(),
            "first_success_step": self._first_success_step.copy(),
            "max_score": self._max_score.copy(),
            "final_score": self._final_score.copy(),
            "distances": self._distances.copy(),
            "n_envs": self.n_envs,
        }
        success = out["per_env_success"].astype(bool)
        if success.any():
            steps = out["first_success_step"][success]
            ttc_steps = float(np.mean(steps))
            out["ttc_steps_mean"] = ttc_steps
            out["ttc_sim_time_mean_s"] = ttc_steps * self._control_timestep_s
        else:
            out["ttc_steps_mean"] = float("nan")
            out["ttc_sim_time_mean_s"] = float("nan")
        out["control_timestep_s"] = self._control_timestep_s
        # DP-paper canonical aggregates: mean over envs of per-env peak / final
        # coverage. Matches official DP repo's 'test/mean_score'.
        out["mean_max_coverage"] = float(np.mean(out["max_score"]))
        out["mean_final_coverage"] = float(np.mean(out["final_score"]))
        # final-frame-stabilized success (un-latched): block in tolerance AT the
        # last step (final), and for all of the last K steps (held). Reported
        # alongside the latched success_rate so drift-after-reach is visible.
        held = (self._held_streak >= self._held_k)
        out["per_env_success_final"] = self._last_success.copy()
        out["success_rate_final"] = float(self._last_success.mean())
        out["per_env_success_held"] = held
        out["success_rate_held"] = float(held.mean())
        out["held_k"] = int(self._held_k)
        return out


# ---------------------------------------------------------------------------
# Cube GTR base + Phase-3 extension (combined)
# ---------------------------------------------------------------------------

class CubeGTRMetrics(TaskMetrics):
    """Per-env Grasp-Transport-Reach tracker with TTC.

    update() reads ``info["behavioral"]`` which must contain:
        cube_pos        : array-like (3,)
        effector_pos    : array-like (3,)
        gripper_opening : float
        terminated      : bool
        goal_pos        : array-like (3,)  — constant per episode
        table_z         : float            — constant per episode

    finalize() output keys
    ----------------------
    success_rate         : float
    grasped              : (n_envs,) bool
    no_drop              : (n_envs,) bool
    reached_goal         : (n_envs,) bool
    success_full         : (n_envs,) bool
    mean_final_dist      : float
    final_dist           : (n_envs,) float
    first_success_step   : (n_envs,) int (>=0 when successful, else -1)
    ttc_steps_mean       : float (mean over successful envs)
    ttc_sim_time_mean_s  : float
    control_timestep_s   : float
    n_envs               : int
    """

    # Canonical thresholds (mirrors latent_eval/success_specs.py _cube_gtr).
    LIFT_THRESH: float = 0.02
    MIN_LIFT_STEPS: int = 3
    HOLD_DIST_THRESH: float = 0.05
    DROP_JUMP_THRESH: float = 0.02
    GOAL_DIST_THRESH: float = 0.07

    def __init__(self, n_envs: int, control_timestep_s: float = 0.05) -> None:
        self.n_envs = n_envs
        self._control_timestep_s = float(control_timestep_s)
        self._states: List[List[dict]] = [[] for _ in range(n_envs)]

    def update(
        self,
        obs: dict,
        info: dict,
        action: np.ndarray,
        step: int,
        env_mask: np.ndarray,
    ) -> None:
        if "behavioral" not in info:
            raise RuntimeError(
                "CubeGTRMetrics.update() requires info['behavioral']. "
                "Got keys: " + str(list(info.keys()))
            )
        beh = info["behavioral"]
        for env_idx in range(self.n_envs):
            if not env_mask[env_idx]:
                continue
            self._states[env_idx].append({
                "cube_pos":        np.asarray(_per_env(beh["cube_pos"], env_idx, self.n_envs), dtype=np.float64),
                "effector_pos":    np.asarray(_per_env(beh["effector_pos"], env_idx, self.n_envs), dtype=np.float64),
                "gripper_opening": float(_per_env(beh["gripper_opening"], env_idx, self.n_envs)),
                "terminated":      bool(_per_env(beh["terminated"], env_idx, self.n_envs)),
                "goal_pos":        np.asarray(_per_env(beh["goal_pos"], env_idx, self.n_envs), dtype=np.float64),
                "table_z":         float(_per_env(beh["table_z"], env_idx, self.n_envs)),
            })

    def _compute_one(self, states: List[dict]) -> dict:
        empty = {
            "grasped": False, "no_drop": False,
            "final_dist": float("inf"), "reached_goal": False,
            "success_full": False,
        }
        if not states:
            return empty

        cube_pos = np.array([s["cube_pos"] for s in states])
        cube_z = cube_pos[:, 2]
        eff_pos = np.array([s["effector_pos"] for s in states])
        eff_cube_dist = np.linalg.norm(eff_pos - cube_pos, axis=1)
        goal_pos = np.asarray(states[-1]["goal_pos"])
        table_z = states[-1]["table_z"]

        lifted = cube_z > (table_z + self.LIFT_THRESH)
        near = eff_cube_dist < self.HOLD_DIST_THRESH
        grasp = lifted & near

        longest = max(
            (sum(1 for _ in g) for k, g in groupby(grasp.tolist()) if k),
            default=0,
        )
        grasped = longest >= self.MIN_LIFT_STEPS
        first_grasp = int(np.argmax(grasp)) if grasp.any() else -1

        no_drop = False
        if grasped and first_grasp >= 0:
            no_drop = True
            ddist = np.abs(np.diff(eff_cube_dist))
            elevation_guard = table_z + 0.03
            placement_proximity = 0.12
            for i in range(first_grasp, len(states) - 1):
                cube_goal_dist = float(np.linalg.norm(cube_pos[i] - goal_pos))
                if cube_goal_dist < placement_proximity:
                    continue
                if ddist[i] > self.DROP_JUMP_THRESH and cube_z[i] > elevation_guard:
                    no_drop = False
                    break

        final_dist = float(np.linalg.norm(cube_pos[-1] - goal_pos))
        reached_goal = final_dist < self.GOAL_DIST_THRESH

        return {
            "grasped": bool(grasped),
            "no_drop": bool(no_drop),
            "final_dist": final_dist,
            "reached_goal": bool(reached_goal),
            "success_full": bool(grasped and no_drop and reached_goal),
        }

    def _first_reached_step(self, states: List[dict]) -> int:
        """First step where cube is within GOAL_DIST_THRESH of goal; -1 if never."""
        if not states:
            return -1
        goal = np.asarray(states[-1]["goal_pos"], dtype=np.float64)
        thresh = self.GOAL_DIST_THRESH
        for i, s in enumerate(states):
            d = float(np.linalg.norm(np.asarray(s["cube_pos"]) - goal))
            if d < thresh:
                return i
        return -1

    def finalize(self) -> dict:
        results = [self._compute_one(self._states[i]) for i in range(self.n_envs)]
        grasped = np.array([r["grasped"] for r in results], dtype=bool)
        no_drop = np.array([r["no_drop"] for r in results], dtype=bool)
        reached_goal = np.array([r["reached_goal"] for r in results], dtype=bool)
        success_full = np.array([r["success_full"] for r in results], dtype=bool)
        final_dist = np.array([r["final_dist"] for r in results], dtype=np.float64)
        out = {
            "success_rate": float(success_full.mean()),
            "grasped": grasped,
            "no_drop": no_drop,
            "reached_goal": reached_goal,
            "success_full": success_full,
            "mean_final_dist": float(final_dist.mean()),
            "final_dist": final_dist,
            "n_envs": self.n_envs,
        }
        # TTC: walk each successful env to find first reached-goal step.
        first_success = np.full(self.n_envs, -1, dtype=np.int64)
        for i in range(self.n_envs):
            if success_full[i]:
                first_success[i] = self._first_reached_step(self._states[i])
        out["first_success_step"] = first_success
        valid = first_success[(first_success >= 0) & success_full]
        if valid.size > 0:
            ttc_steps = float(np.mean(valid))
            out["ttc_steps_mean"] = ttc_steps
            out["ttc_sim_time_mean_s"] = ttc_steps * self._control_timestep_s
        else:
            out["ttc_steps_mean"] = float("nan")
            out["ttc_sim_time_mean_s"] = float("nan")
        out["control_timestep_s"] = self._control_timestep_s
        return out


# ---------------------------------------------------------------------------
# Granular (PyFleX) chamfer-to-goal metrics — mirrors PushTCoverageMetrics
# but tracks chamfer distance instead of coverage IoU. Success is CD <
# success_threshold; lower chamfer is better, so we track min not max.
# ---------------------------------------------------------------------------

class GranularChamferMetrics(TaskMetrics):
    """Per-env chamfer-to-goal tracker for the swm/Granular-v0 task.

    finalize() output keys
    ----------------------
    success_rate          : float       — fraction of envs that reached CD<thresh
    per_env_success       : (n_envs,) bool
    first_success_step    : (n_envs,) int   — timeout_penalty_step on failure
    min_chamfer           : (n_envs,) float — best (lowest) CD over trajectory
    final_chamfer         : (n_envs,) float — CD at last step recorded
    init_chamfer          : (n_envs,) float — CD at step 0
    chamfer_trajectory    : (n_envs, eval_budget) float
    n_envs                : int
    ttc_steps_mean        : float (mean over successful envs)
    ttc_sim_time_mean_s   : float (= ttc_steps_mean * control_timestep_s)
    control_timestep_s    : float
    mean_init_chamfer     : float
    mean_final_chamfer    : float
    mean_min_chamfer      : float
    """

    def __init__(
        self,
        n_envs: int,
        eval_budget: int,
        control_timestep_s: float = 0.5,
        success_threshold: float = 0.03,
        timeout_penalty_step: Optional[int] = None,
    ) -> None:
        self.n_envs = n_envs
        self.eval_budget = eval_budget
        self._control_timestep_s = float(control_timestep_s)
        self._success_threshold = float(success_threshold)
        self.timeout_penalty_step: int = (
            timeout_penalty_step if timeout_penalty_step is not None else eval_budget
        )
        self._success: np.ndarray = np.zeros(n_envs, dtype=bool)
        self._first_success_step: np.ndarray = np.full(
            n_envs, self.timeout_penalty_step, dtype=np.int64
        )
        # Initialise min/final to +inf / 0 so the first update() seeds them.
        self._min_chamfer: np.ndarray = np.full(n_envs, np.inf, dtype=np.float64)
        self._init_chamfer: np.ndarray = np.full(n_envs, np.nan, dtype=np.float64)
        self._final_chamfer: np.ndarray = np.zeros(n_envs, dtype=np.float64)
        self._chamfer_traj: np.ndarray = np.full(
            (n_envs, eval_budget), np.nan, dtype=np.float64
        )

    def _extract_chamfer(self, info: dict, env_idx: int) -> float:
        """Read per-env chamfer-to-goal. Falls back to NaN if absent."""
        if "chamfer" in info:
            return float(_per_env(info["chamfer"], env_idx, self.n_envs))
        return float("nan")

    def _extract_success(self, info: dict, env_idx: int) -> bool:
        """Prefer pre-computed info['success']; else derive from chamfer < threshold."""
        if "success" in info:
            return bool(_per_env(info["success"], env_idx, self.n_envs))
        cd = self._extract_chamfer(info, env_idx)
        return bool(np.isfinite(cd) and cd < self._success_threshold)

    def update(
        self,
        obs: dict,
        info: dict,
        action: np.ndarray,
        step: int,
        env_mask: np.ndarray,
    ) -> None:
        col = min(step, self.eval_budget - 1)
        for env_idx in range(self.n_envs):
            if not env_mask[env_idx]:
                continue
            cd = self._extract_chamfer(info, env_idx)
            self._chamfer_traj[env_idx, col] = cd
            if np.isnan(self._init_chamfer[env_idx]) and np.isfinite(cd):
                self._init_chamfer[env_idx] = cd
            if np.isfinite(cd) and cd < self._min_chamfer[env_idx]:
                self._min_chamfer[env_idx] = cd
            if np.isfinite(cd):
                self._final_chamfer[env_idx] = cd
            if not self._success[env_idx] and self._extract_success(info, env_idx):
                self._success[env_idx] = True
                self._first_success_step[env_idx] = step

    def finalize(self) -> dict:
        # Replace any leftover +inf in min_chamfer (env_mask never True) with NaN.
        min_cd = np.where(np.isinf(self._min_chamfer), np.nan, self._min_chamfer)
        out = {
            "success_rate": float(self._success.mean()),
            "per_env_success": self._success.copy(),
            "first_success_step": self._first_success_step.copy(),
            "min_chamfer": min_cd.copy(),
            "final_chamfer": self._final_chamfer.copy(),
            "init_chamfer": self._init_chamfer.copy(),
            "chamfer_trajectory": self._chamfer_traj.copy(),
            "n_envs": self.n_envs,
        }
        success = out["per_env_success"].astype(bool)
        if success.any():
            steps = out["first_success_step"][success]
            ttc_steps = float(np.mean(steps))
            out["ttc_steps_mean"] = ttc_steps
            out["ttc_sim_time_mean_s"] = ttc_steps * self._control_timestep_s
        else:
            out["ttc_steps_mean"] = float("nan")
            out["ttc_sim_time_mean_s"] = float("nan")
        out["control_timestep_s"] = self._control_timestep_s
        out["success_threshold"] = self._success_threshold
        # Mean aggregates (NaN-safe — uses nanmean so envs with no data
        # do not poison the average).
        out["mean_init_chamfer"]  = float(np.nanmean(out["init_chamfer"]))
        out["mean_final_chamfer"] = float(np.nanmean(out["final_chamfer"]))
        out["mean_min_chamfer"]   = float(np.nanmean(out["min_chamfer"]))
        return out
