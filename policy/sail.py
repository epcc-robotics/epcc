"""SAIL policies — faster-than-demonstration execution (arXiv:2506.11948).

Both classes speed execution up the same way the paper does: shrink the env's
``control_timestep`` to ``delta * c`` so the action stream is consumed on a
compressed time axis. The caller's recipe scales the eval budget by the same
factor to hold the simulated-time horizon constant.

``ReachedPoseSAILPolicy`` is the policy we run.
    Its inner DP predicts absolute *reached* EE poses (paper 4.2,
    controller-invariant targets) and the wrapper tracks them closed-loop
    against the live env pose. One flag switches the two configurations we
    evaluate:
        use_precision=False -> SAIL without the AWE scheduler (always fast)
        use_precision=True  -> SAIL with it (precision bit toggles slow/fast)

``SAILPolicy`` is the naive baseline it is measured against.
    Control-rate scaling only, inner actions passed through untouched — this
    is the paper's DP-Fast. Kept because every proper-SAIL claim is stated
    relative to it.

See ``scripts/sail/`` for the label -> train -> evaluate pipeline and
``findings/260716_sail_improve/report.md`` for measured behaviour.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from policy.base import TaskOrthogonalPolicy, WrapperPolicy


def _resolve_timesteps(
    base_control_timestep: float,
    base_physics_timestep: float,
    dt_factor: float,
) -> tuple[float, float]:
    """Mirror ``policy.sail._resolve_timesteps`` — guarantee integer substeps."""
    target_control = base_control_timestep / dt_factor
    n_substeps_float = target_control / base_physics_timestep
    n_substeps = max(1, int(round(n_substeps_float)))
    physics = target_control / n_substeps
    return target_control, physics


def _apply_timesteps(
    env_unwrapped: Any,
    control_timestep: float,
    physics_timestep: float,
    exact: bool = True,
) -> None:
    """Push a resolved (control, physics) timestep pair onto one env.

    OGBench ManipSpace takes both through ``set_timesteps``. PushT2D instead
    carries ``dt`` (physics) and ``control_hz`` separately and derives its
    substep count from their product, so moving ``control_hz`` alone quantises
    the requested speed-up — k=3 silently executed at k=3.33 because
    ``round(1/(0.01*30)) == 3`` spans 0.030 s, not 0.0333 s. Setting ``dt`` too
    makes the product land on an exact integer.

    ``exact=False`` keeps the old control_hz-only path. Note it is not a
    bit-exact replay of pre-fix runs: the env now rounds its substep count
    instead of truncating, which shifts k values whose quotient was not already
    near-integer (k=1.5 -> 7 substeps, previously 6).
    """
    if hasattr(env_unwrapped, "set_timesteps"):
        env_unwrapped.set_timesteps(
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
        )
        return
    if hasattr(env_unwrapped, "control_hz"):
        if exact and hasattr(env_unwrapped, "dt"):
            env_unwrapped.dt = physics_timestep
        env_unwrapped.control_hz = 1.0 / control_timestep


class SAILPolicy(WrapperPolicy):
    """Wraps any TaskOrthogonalPolicy + scales env control rate by dt_factor.

    Construction kwargs:
      inner                  : TaskOrthogonalPolicy — the underlying policy
      dt_factor              : float — speed-up factor k (k>1.0 = faster)
      base_control_timestep  : float — env's default control_timestep (s)
      base_physics_timestep  : float — env's default physics_timestep (s)
    """

    def __init__(
        self,
        inner: TaskOrthogonalPolicy,
        dt_factor: float = 2.5,
        base_control_timestep: float = 0.05,
        base_physics_timestep: float = 0.002,
        exact_timesteps: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(inner=inner, **kwargs)
        self._dt_factor = float(dt_factor)
        self._base_control_timestep = float(base_control_timestep)
        self._base_physics_timestep = float(base_physics_timestep)
        self._exact_timesteps = bool(exact_timesteps)

    @property
    def dt_factor(self) -> float:
        return self._dt_factor

    @property
    def control_timestep(self) -> float:
        ts, _ = _resolve_timesteps(
            self._base_control_timestep, self._base_physics_timestep, self._dt_factor
        )
        return ts

    def reset(self, world: Any, cfg: Any | None = None, eval_world: Any | None = None) -> None:
        """Scale env control_timestep, then delegate to inner.reset()."""
        control_ts, physics_ts = _resolve_timesteps(
            self._base_control_timestep, self._base_physics_timestep, self._dt_factor
        )
        for env in world.envs.unwrapped.envs:
            _apply_timesteps(env.unwrapped, control_ts, physics_ts, self._exact_timesteps)
        # Forward all reset args to inner; degrade gracefully for simpler subclasses.
        for attempt in (
            lambda: self._inner.reset(world, cfg=cfg, eval_world=eval_world),
            lambda: self._inner.reset(world, cfg=cfg),
            lambda: self._inner.reset(world),
        ):
            try:
                return attempt()
            except TypeError:
                continue
        raise RuntimeError(
            f"SAILPolicy inner {type(self._inner).__name__}.reset() rejected all signatures"
        )

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        return self._inner.get_action(info_dict, **kwargs)


class ReachedPoseSAILPolicy(WrapperPolicy):
    """Proper SAIL execution (paper §4.2 + §4.3) for OGBench ManipSpace envs.

    Inner policy predicts the ABSOLUTE reached EE pose (controller-invariant
    target, FastIL ``absolute_actions_with_precision`` convention):
    ``(n_envs, 6) = [x, y, z, yaw, gripper_cmd, precision]``.

    Each ``get_action`` this wrapper:
      1. Converts the absolute pose target into a per-step relative action
         against the LIVE env pose: ``delta = clip((target - current)/range)``.
         This closes the tracking loop — lagging envs get larger deltas —
         which is what makes faster-than-demo execution survivable.
      2. Reads the precision bit and toggles ``slow_dt_factor`` /
         ``fast_dt_factor``. ``use_precision=False`` disables this (always
         fast) — that is the "SAIL without AWE scheduler" arm.
      3. Returns the 5-D env action (precision stripped).

    Replay-validated ceiling (findings/260716_sail_improve): tracking demo
    reached poses this way gives 100% GTR at k=2.5, 94% at k=3.0 (stock gains).

    Precision scheduling (paper 4.3, AWE). ``precision_mode`` selects how the
    per-env precision bits become control rates:

    ``"per_env"`` (default) gives every env its own rate. This is possible
    because the eval builds a ``SyncVectorEnv`` (stable_worldmodel/world.py),
    i.e. 50 ordinary in-process objects stepped serially — ``_apply_timesteps``
    writes instance attributes and takes effect on that env's next ``step()``.
    An earlier note here claimed the batch shares one ``control_timestep`` and
    that proving 4.3 needed a single-env lane; that was wrong.

    ``"majority"`` is the legacy path, kept only to replay archived cells: one
    vote across the batch sets one rate for everybody. It is provably inert at
    n=50 — precision frames are 4.3% (cube) / 14.2% (plain) / 25.9% (slippery)
    of the labels and the 50 episodes are phase-desynced, so the mean never
    reaches ``precision_majority`` and the rate stays pinned at fast.
    """

    #: per-step action range — mirrors ogbench manipspace_env.py action_range.
    ACTION_RANGE = np.array([0.05, 0.05, 0.05, 0.3, 1.0])

    def __init__(
        self,
        inner: TaskOrthogonalPolicy,
        fast_dt_factor: float = 2.5,
        slow_dt_factor: float = 1.0,
        base_control_timestep: float = 0.05,
        base_physics_timestep: float = 0.002,
        env_action_dim: int = 5,
        use_precision: bool = True,
        precision_threshold: float = 0.5,
        precision_majority: float = 0.5,
        precision_mode: str = "per_env",
        exact_timesteps: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(inner=inner, **kwargs)
        self._fast_dt_factor = float(fast_dt_factor)
        self._slow_dt_factor = float(slow_dt_factor)
        self._base_control_timestep = float(base_control_timestep)
        self._base_physics_timestep = float(base_physics_timestep)
        self._exact_timesteps = bool(exact_timesteps)
        self._env_action_dim = int(env_action_dim)
        self._use_precision = bool(use_precision)
        self._precision_threshold = float(precision_threshold)
        self._precision_majority = float(precision_majority)
        if precision_mode not in ("per_env", "majority"):
            raise ValueError(f"precision_mode must be 'per_env' or 'majority', got {precision_mode!r}")
        self._precision_mode = str(precision_mode)
        self._world: Any | None = None
        # Per-env cache of the factor last written to each env. NaN never
        # compares equal, so the first _apply_dt always writes every env —
        # reproducing the old `None` sentinel exactly.
        self._current_dt_factors: np.ndarray | None = None
        self._ts_cache: dict[float, tuple[float, float]] = {}
        self._precision_trace: list[np.ndarray] = []

    @property
    def current_dt_factor(self) -> float | None:
        """The batch-wide factor, or None once the envs disagree."""
        f = self._current_dt_factors
        if f is None or len(f) == 0 or not np.all(f == f[0]):
            return None
        return float(f[0])

    @property
    def current_dt_factors(self) -> np.ndarray | None:
        return self._current_dt_factors

    @property
    def precision_bits_trace(self) -> np.ndarray | None:
        """(T, n_envs) bool — what the DP's precision head asked for each step.

        Distinguishes "the scheduler is inert" from "the head never fires",
        which the realised control_ts alone cannot.
        """
        return np.stack(self._precision_trace) if self._precision_trace else None

    def _resolve_cached(self, dt_factor: float) -> tuple[float, float]:
        f = float(dt_factor)
        if f not in self._ts_cache:
            self._ts_cache[f] = _resolve_timesteps(
                self._base_control_timestep, self._base_physics_timestep, f
            )
        return self._ts_cache[f]

    def _apply_dt(self, dt_factor: "float | np.ndarray") -> None:
        """Write the control rate. Accepts a scalar or an (n_envs,) array.

        The scalar path is what `reset()` and the constant-k `SAILPolicy` use;
        broadcasting it against a NaN-seeded cache issues exactly the same
        per-env writes, in the same order, as the previous global loop.
        """
        if self._world is None:
            return
        envs = self._world.envs.unwrapped.envs
        n = len(envs)
        if self._current_dt_factors is None or len(self._current_dt_factors) != n:
            self._current_dt_factors = np.full(n, np.nan, dtype=np.float64)
        req = np.broadcast_to(np.asarray(dt_factor, dtype=np.float64), (n,))
        dirty = ~(req == self._current_dt_factors)   # NaN != anything -> first pass writes all
        if not dirty.any():
            return
        for i in np.flatnonzero(dirty):
            control_ts, physics_ts = self._resolve_cached(req[i])
            _apply_timesteps(envs[i].unwrapped, control_ts, physics_ts, self._exact_timesteps)
        self._current_dt_factors[dirty] = req[dirty]

    def _schedule_precision(self, action: np.ndarray) -> None:
        """Turn the DP's precision channel into control rates.

        Single implementation for both ReachedPose policies — it used to be
        duplicated verbatim in sail_pusht.py, so a fix applied here alone would
        silently have missed the two tasks with the most precision frames.
        """
        if not self._use_precision:
            return
        bits = np.asarray(action[:, -1]) > self._precision_threshold
        self._precision_trace.append(bits.copy())
        if self._precision_mode == "majority":
            self._apply_dt(self._slow_dt_factor if float(bits.mean()) >= self._precision_majority
                           else self._fast_dt_factor)
        else:
            self._apply_dt(np.where(bits, self._slow_dt_factor, self._fast_dt_factor))

    @staticmethod
    def _live_pose(ev: Any) -> tuple[np.ndarray, float]:
        from ogbench.manipspace import lie

        pos = ev._data.site_xpos[ev._pinch_site_id].copy()
        yaw = lie.SO3.from_matrix(
            ev._data.site_xmat[ev._pinch_site_id].copy().reshape(3, 3)
        ).compute_yaw_radians()
        return pos, float(yaw)

    def reset(self, world: Any, cfg: Any | None = None, eval_world: Any | None = None) -> None:
        self._world = world
        self._current_dt_factors = None
        self._ts_cache = {}
        self._precision_trace = []
        self._apply_dt(self._fast_dt_factor)
        print(f"[sail] precision_mode={self._precision_mode} use_precision={self._use_precision} "
              f"fast={self._fast_dt_factor} slow={self._slow_dt_factor}")
        for attempt in (
            lambda: self._inner.reset(world, cfg=cfg, eval_world=eval_world),
            lambda: self._inner.reset(world, cfg=cfg),
            lambda: self._inner.reset(world),
        ):
            try:
                return attempt()
            except TypeError:
                continue
        raise RuntimeError(
            f"ReachedPoseSAILPolicy inner {type(self._inner).__name__}.reset() "
            f"rejected all signatures"
        )

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        action = np.asarray(self._inner.get_action(info_dict, **kwargs), dtype=np.float64)
        expected_dim = self._env_action_dim + 1
        if action.shape[-1] != expected_dim:
            raise ValueError(
                f"ReachedPoseSAILPolicy: inner returned action dim "
                f"{action.shape[-1]}, expected {expected_dim} "
                f"(absolute pose {self._env_action_dim} + 1 precision)"
            )
        action = np.atleast_2d(action)

        self._schedule_precision(action)   # no-op when use_precision is off

        out = np.zeros((action.shape[0], self._env_action_dim), dtype=np.float32)
        envs = self._world.envs.unwrapped.envs
        for i in range(action.shape[0]):
            ev = envs[i].unwrapped
            pos, yaw = self._live_pose(ev)
            out[i, :3] = np.clip(
                (action[i, :3] - pos) / self.ACTION_RANGE[:3], -1.0, 1.0
            )
            dyaw = (action[i, 3] - yaw + np.pi) % (2 * np.pi) - np.pi
            out[i, 3] = np.clip(dyaw / self.ACTION_RANGE[3], -1.0, 1.0)
            out[i, 4] = np.clip(action[i, 4], -1.0, 1.0)  # gripper stays commanded
        return out
