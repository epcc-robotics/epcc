"""SkiP execution wrapper for our cube pipeline.

The inner DP is trained (via CubeAWSkipImageDataset) to output ABSOLUTE ee
poses [xyz(3), yaw(1), gripper(1)] with SkiP relabel (skip steps -> next-key
pose = a leap). This wrapper converts that absolute target into the env's
DELTA action:  a_norm = clip((abs_target - current_state) / action_range, -1, 1).

Because OGBench's per-step action range is generous (xyz = 0.05 m/step vs the
demo's ~0.005 m/step at c=0.3), a leap of a few cm is covered in ~1-2 steps at
the env's max speed — so skip regions execute fast, key regions (target near
current) stay small/precise. No motion planner, no env modification.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from policy.base import WrapperPolicy

# OGBench manipspace action_range (manipspace_env.py:114) — physical per-step.
ACTION_RANGE = np.array([0.05, 0.05, 0.05, 0.3, 1.0], dtype=np.float64)


class SkipPolicy(WrapperPolicy):
    """Wrap an absolute-pose DP; emit delta actions toward the predicted pose."""

    def __init__(self, inner, **kwargs: Any) -> None:
        super().__init__(inner=inner, **kwargs)
        self._world = None

    def reset(self, world: Any, cfg: Any | None = None, eval_world: Any | None = None) -> None:
        self._world = world
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
            f"SkipPolicy inner {type(self._inner).__name__}.reset() rejected all signatures")

    def _current_state(self):
        """(n_envs, 5) current [eff_xyz, eff_yaw, gripper_opening] per env."""
        rows = []
        for env in self._world.envs.unwrapped.envs:
            info = env.unwrapped.compute_ob_info()
            rows.append(np.concatenate([
                np.asarray(info["proprio/effector_pos"], float).reshape(3),
                np.asarray(info["proprio/effector_yaw"], float).reshape(1),
                np.asarray(info["proprio/gripper_opening"], float).reshape(1),
            ]))
        return np.stack(rows, axis=0)

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        abs_target = np.asarray(self._inner.get_action(info_dict, **kwargs), float)  # (n,5) abs pose
        cur = self._current_state()                                                  # (n,5)
        if abs_target.shape != cur.shape:
            raise ValueError(f"SkipPolicy: DP output {abs_target.shape} != state {cur.shape} "
                             f"(inner must be an abs-pose DP, action_dim=5)")
        delta = abs_target - cur
        a_norm = np.clip(delta / ACTION_RANGE, -1.0, 1.0)
        return a_norm.astype(np.float32)


class PushTSkipPolicy(SkipPolicy):
    """SkiP wrapper for PushT (2-d agent-pos).

    Inner DP (PushTSVSkipImageDataset) outputs ABSOLUTE agent xy (pixels) with
    SkiP relabel. The PushT env applies (env.py:345, relative mode):
        env_target = agent.position + action * action_scale
    so the env-action that lands the agent on ``abs_target`` is
        a = clip((abs_target - agent.position) / action_scale, -1, 1).
    Mirrors the cube SkipPolicy exactly, with ACTION_RANGE -> env.action_scale
    and current state -> agent.position (2-d). reset() is inherited.
    """

    def _agent_state(self):
        """((n,2) agent xy, (n,1) action_scale) per env."""
        rows, scales = [], []
        for env in self._world.envs.unwrapped.envs:
            u = env.unwrapped
            rows.append(np.asarray(u.agent.position, float).reshape(2))
            scales.append(float(getattr(u, "action_scale", 100.0)))
        return np.stack(rows, axis=0), np.asarray(scales, float).reshape(-1, 1)

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        abs_target = np.asarray(self._inner.get_action(info_dict, **kwargs), float)  # (n,2) abs xy
        cur, scale = self._agent_state()                                             # (n,2), (n,1)
        if abs_target.shape != cur.shape:
            raise ValueError(f"PushTSkipPolicy: DP output {abs_target.shape} != agent-pos "
                             f"{cur.shape} (inner must be an abs-agent-pos DP, action_dim=2)")
        a_norm = np.clip((abs_target - cur) / scale, -1.0, 1.0)
        return a_norm.astype(np.float32)
