"""Reached-pose SAIL for the PushT family.

``policy.sail.ReachedPoseSAILPolicy`` is the real SAIL body (paper §4.2/§4.3)
but it is written against OGBench ManipSpace: it reads the live pose from
``env._data.site_xpos[env._pinch_site_id]``, imports ``ogbench.manipspace.lie``
for the yaw, and emits a 5-D [x,y,z,yaw,grip] action. PushT has a 2-D agent
position, no orientation and no gripper, so the geometry has to be replaced —
but the mechanism that makes SAIL work does not.

That mechanism is the closed tracking loop: the inner DP predicts an ABSOLUTE
reached pose, and the wrapper converts it to a relative command against the
LIVE pose every step, so an env that has fallen behind is handed a larger
delta. Open-loop speed-up (``SAILPolicy``, the paper's DP-Fast) has no such
feedback and collapses as k grows — 36% vs 90% at k=2.5 on cube.

``policy/sail.py`` is not modified; this subclasses it.

🔑 ACTION_RANGE = 100 px is not a fitted constant. PushT's env computes
``action = agent.position + action * self.action_scale`` with
``action_scale = 100`` (``envs/pusht/env.py:65,:360``), so dividing the
positional error by 100 is the exact inverse of what the env will do. Verified
closed-loop (job 11602468): a 120 px error falls to 0.14 px in 10 control steps
and 0.00 by 40. The separate ``waypoint_scale ~= 39.96`` that SuP uses is a
different quantity — px actually ACHIEVED in one control step, which is smaller
because the PD loop only closes part of the gap per step.

🔑 base timesteps are PushT's, not cube's: control 1/control_hz = 0.1 s and
physics dt = 0.01 s, against ManipSpace's 0.05 / 0.002. Getting these wrong
silently rescales every k in the sweep.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from policy.sail import ReachedPoseSAILPolicy


class ReachedPosePushTSAILPolicy(ReachedPoseSAILPolicy):
    """Inner policy emits (n_envs, 3) = [x, y, precision] in PIXELS."""

    #: px per unit env action — exactly PushT's `action_scale`.
    ACTION_RANGE = np.array([100.0, 100.0])

    def __init__(
        self,
        inner: Any,
        fast_dt_factor: float = 2.5,
        slow_dt_factor: float = 1.0,
        base_control_timestep: float = 0.1,    # PushT: 1 / control_hz(10)
        base_physics_timestep: float = 0.01,   # PushT: dt
        env_action_dim: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            inner=inner,
            fast_dt_factor=fast_dt_factor,
            slow_dt_factor=slow_dt_factor,
            base_control_timestep=base_control_timestep,
            base_physics_timestep=base_physics_timestep,
            env_action_dim=env_action_dim,
            **kwargs,
        )

    @staticmethod
    def _live_pose(ev: Any):
        """PushT agent position. No orientation channel exists."""
        return np.asarray(ev.agent.position, dtype=np.float64), 0.0

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        action = np.asarray(self._inner.get_action(info_dict, **kwargs), dtype=np.float64)
        expected_dim = self._env_action_dim + 1
        if action.shape[-1] != expected_dim:
            raise ValueError(
                f"ReachedPosePushTSAILPolicy: inner returned action dim "
                f"{action.shape[-1]}, expected {expected_dim} "
                f"(absolute pose {self._env_action_dim} + 1 precision). The "
                f"reached-pose DP must be trained on the 3-D `action` key that "
                f"scripts/sail/make_labels_pusht.py writes."
            )
        action = np.atleast_2d(action)

        # Was a verbatim copy of the parent's majority vote. Hoisted to
        # ReachedPoseSAILPolicy._schedule_precision so per-env scheduling has
        # exactly one implementation — PushT is where it matters most (14.2%
        # plain / 25.9% slippery precision frames vs cube's 4.3%).
        self._schedule_precision(action)   # no-op when use_precision is off

        envs = self._world.envs.unwrapped.envs
        out = np.zeros((action.shape[0], self._env_action_dim), dtype=np.float32)
        for i in range(action.shape[0]):
            pos, _ = self._live_pose(envs[i].unwrapped)
            out[i] = np.clip(
                (action[i, : self._env_action_dim] - pos) / self.ACTION_RANGE, -1.0, 1.0
            )
        return out
