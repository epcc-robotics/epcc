"""Narrow-init variation_space patch for swm/PushT-v1.

Mutates `variation_space['block']['start_position'].low/high` and
`variation_space['block']['angle'].low/high` on every sub-env so subsequent
`env.reset()` calls sample inits from the narrow box. Optionally also tightens
agent start position.

Mechanism: `swm_spaces.reset_variation_space(...)` (in
stable_worldmodel/envs/pusht/env.py:257) samples using the *current* `low`/`high`
fields of each `Box`, so mutating them in place is enough.

Usage:
    from scripts.data._pusht_narrow_init_patch import patch_pusht_narrow_init

    world = swm.World("swm/PushT-v1", num_envs=N, ...)
    patch_pusht_narrow_init(
        world,
        block_xy_low=[385, 85],
        block_xy_high=[415, 115],
        block_angle_low=0.6981,    # π/4 - π/36 ≈ 40°
        block_angle_high=0.8727,   # π/4 + π/36 ≈ 50°
    )
"""
from __future__ import annotations
from typing import Optional, Sequence
import numpy as np


def patch_pusht_narrow_init(
    world,
    block_xy_low: Sequence[float],
    block_xy_high: Sequence[float],
    block_angle_low: float,
    block_angle_high: float,
    agent_xy: Optional[tuple[Sequence[float], Sequence[float]]] = None,
) -> None:
    """Mutate variation_space bounds on every sub-env in `world`.

    Args:
        world: swm.World instance (already constructed).
        block_xy_low: [x_low, y_low] for block start position, pixel coords ∈ [0, 512].
        block_xy_high: [x_high, y_high] for block start position.
        block_angle_low: radians.
        block_angle_high: radians.
        agent_xy: optional ([x_low, y_low], [x_high, y_high]) for agent start position.

    After calling, every subsequent `env.reset(seed=...)` on any sub-env will
    sample inits from the narrow box.
    """
    block_xy_low_arr = np.asarray(block_xy_low, dtype=np.float64)
    block_xy_high_arr = np.asarray(block_xy_high, dtype=np.float64)
    assert block_xy_low_arr.shape == (2,), f"block_xy_low must be 2-D, got {block_xy_low_arr.shape}"
    assert block_xy_high_arr.shape == (2,), f"block_xy_high must be 2-D, got {block_xy_high_arr.shape}"
    assert (block_xy_low_arr <= block_xy_high_arr).all(), \
        f"block_xy_low {block_xy_low_arr} must be <= block_xy_high {block_xy_high_arr}"
    assert block_angle_low <= block_angle_high, \
        f"block_angle_low {block_angle_low} must be <= block_angle_high {block_angle_high}"

    for env in world.envs.unwrapped.envs:
        base = env
        while hasattr(base, "env"):
            base = base.env
        vs = base.variation_space

        vs['block']['start_position'].low = block_xy_low_arr.copy()
        vs['block']['start_position'].high = block_xy_high_arr.copy()
        # block.angle is a 0-D Box (shape=()); sampler indexes low/high so we need 0-D ndarrays, not Python floats
        vs['block']['angle'].low = np.asarray(float(block_angle_low), dtype=np.float64)
        vs['block']['angle'].high = np.asarray(float(block_angle_high), dtype=np.float64)

        if agent_xy is not None:
            agent_low_arr = np.asarray(agent_xy[0], dtype=np.float64)
            agent_high_arr = np.asarray(agent_xy[1], dtype=np.float64)
            assert agent_low_arr.shape == (2,)
            assert agent_high_arr.shape == (2,)
            assert (agent_low_arr <= agent_high_arr).all()
            vs['agent']['start_position'].low = agent_low_arr.copy()
            vs['agent']['start_position'].high = agent_high_arr.copy()
