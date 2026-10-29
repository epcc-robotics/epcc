"""Task-orthogonal policy base classes for the Phase 3 unified eval loop.

Hierarchy:
  TaskOrthogonalPolicy(swm.BasePolicy)
    ├─ ChunkedReplanPolicy   — sample a chunk of actions, consume one/step,
    │                          replan every N consumed (DP, CleanDiffuser-DP,
    │                          DTG+MPCC, DTG+InvDyn)
    ├─ SolverPolicy          — solve an inner optimization per step with
    │                          warm-start across steps (MPCC, WM-Full)
    └─ WrapperPolicy         — composes another policy + an env tweak
                               (SAIL = inner + control_timestep scaling)

The unified loop calls ``policy.get_action(info_dict)`` once per env-step
and gets one action back. Internal state (queues, solver warm-starts,
replan counters, CEM means) lives in ``self``, never in the caller.

Per-task config (env name, obs shape, action dim, metric class) lives in
Hydra recipes — never in the policy class.
"""
from __future__ import annotations

import abc
from abc import abstractmethod
from collections import deque
from typing import Any

import numpy as np
import stable_worldmodel as swm


class TaskOrthogonalPolicy(swm.policy.BasePolicy, abc.ABC):
    """``swm.BasePolicy`` with an explicit ``reset(world)`` hook + ABC enforcement.

    The unified loop calls ``reset(world)`` once after
    ``world.set_policy(self)`` and before the first ``world.step()``.
    """

    def reset(self, world: Any) -> None:
        """Re-initialize per-rollout state. Default no-op; override to clear
        action queues / solver warm-starts."""
        return


class ChunkedReplanPolicy(TaskOrthogonalPolicy):
    """Policies that batch-sample a chunk of actions, consume one per step,
    and re-sample every ``replan_after_n`` consumed actions.

    Concrete subclasses implement ``_predict_chunk(info_dict) -> (K, n_envs,
    action_dim)`` where K is the chunk length. The base class owns the queue
    + replan-cadence state.

    Used by DP / CleanDiffuser-DP (chunk=16, replan after 8),
    DTG+MPCC / DTG+InvDyn (chunk=K, replan after K).
    """

    def __init__(self, chunk_size: int, replan_after_n: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if replan_after_n > chunk_size:
            raise ValueError(
                f"replan_after_n ({replan_after_n}) > chunk_size ({chunk_size}) "
                f"— would consume actions beyond the chunk."
            )
        self._chunk_size = int(chunk_size)
        self._replan_after = int(replan_after_n)
        self._queue: deque = deque()
        self._consumed_since_replan = 0
        self._n_replans = 0

    def reset(self, world: Any) -> None:
        self._queue = deque()
        self._consumed_since_replan = 0
        self._n_replans = 0

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        if not self._queue or self._consumed_since_replan >= self._replan_after:
            chunk = self._predict_chunk(info_dict)
            # chunk shape: (K, n_envs, action_dim) — push each timestep into queue.
            self._queue = deque(chunk[: self._chunk_size])
            self._consumed_since_replan = 0
            self._n_replans += 1
        action = self._queue.popleft()
        self._consumed_since_replan += 1
        return np.asarray(action)

    @property
    def n_replans(self) -> int:
        """Total number of chunk refills (=replans) since the last reset()."""
        return int(self._n_replans)

    @abstractmethod
    def _predict_chunk(self, info_dict: dict) -> np.ndarray:
        """Return ``(chunk_size, n_envs, action_dim)`` action chunk."""
        ...


class SolverPolicy(TaskOrthogonalPolicy):
    """Policies that solve an inner optimization per env-step.

    Subclasses implement ``get_action(info_dict)`` directly and maintain
    solver warm-start state (e.g. ``self._prev_solution``,
    ``self._cem_mean``) across calls.

    Used by MPCC (per-step QP with warm-start) and WM-Full (CEM in WM with
    warm-start).
    """

    @abstractmethod
    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        ...


class WrapperPolicy(TaskOrthogonalPolicy):
    """Policies that compose an inner ``TaskOrthogonalPolicy`` plus an
    environment tweak applied in ``reset(world)``.

    Used by SAIL: inner = DPPolicy; reset() scales env control_timestep by
    ``dt_factor``; get_action delegates to inner.
    """

    def __init__(self, inner: TaskOrthogonalPolicy, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._inner = inner

    def reset(self, world: Any) -> None:
        self._inner.reset(world)

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        return self._inner.get_action(info_dict, **kwargs)

    @property
    def n_replans(self) -> int:
        """Forward the inner policy's replan count so video overlays see it."""
        return int(getattr(self._inner, "n_replans", 0))
