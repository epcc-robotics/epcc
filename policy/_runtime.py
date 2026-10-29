"""Shared in-process runtime helpers for the Phase 3 unified eval loop.

These helpers re-implement the env-reset + behavioral-extraction logic that
``eval_mpc_multiview.evaluate_tracking_from_dataset`` performs, but as
free functions that the unified loop in ``scripts/baseline_v2/eval_unified.py``
can compose. The legacy file remains untouched (additive rule).

The intentional duplication is the cost of additive-only refactoring; once
all 6 policies are migrated, a follow-up iteration can refactor the shared
piece into ``swm`` and have both v1 and v2 import it.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

import stable_worldmodel as swm

# Extend Hydra's config search path to include the legacy ``config/eval/``
# directory so v2 recipes can reference ``/solver: mpcc_multiview`` etc.
# without duplicating the legacy solver yamls into baseline_corl_v2/.
# Registered at import time so both eval_unified.py and pytest pick it up.
_REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    from hydra.core.plugins import Plugins
    from hydra.core.config_search_path import ConfigSearchPath
    from hydra.plugins.search_path_plugin import SearchPathPlugin

    class _LegacyEvalSearchPath(SearchPathPlugin):
        def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
            search_path.append(
                provider="legacy-eval",
                path=f"file://{_REPO_ROOT}/config/eval",
            )

    Plugins.instance().register(_LegacyEvalSearchPath)
except Exception:
    pass  # Hydra not installed yet or import failed — caller will see clean error.


# ---------------------------------------------------------------------------
# Phase 1: world + dataset + episode selection
# ---------------------------------------------------------------------------

def setup_world_and_dataset(cfg: DictConfig) -> SimpleNamespace:
    """Delegate to the legacy ``_setup_eval_world(cfg)`` for byte-equivalent
    dataset loading, episode selection, transforms, and scaler fitting.

    Returns the SimpleNamespace the legacy helper produces, which has
    attributes: world, dataset, eval_episodes, eval_start_idx, n_eval,
    eval_budget, goal_offset, goal_offsets, scaler_process.
    """
    from eval_mpc_multiview import _setup_eval_world
    return _setup_eval_world(cfg)


# ---------------------------------------------------------------------------
# Phase 2: env reset to dataset chunk
# ---------------------------------------------------------------------------

def reset_world_to_dataset(
    world: Any,
    dataset: Any,
    episode_indices: np.ndarray,
    start_steps: np.ndarray,
    goal_offset_steps: np.ndarray,
    callables: list | None = None,
    init_perturb_fn=None,
    verify_init: bool = False,
) -> tuple[dict, dict]:
    """Reset world envs to the dataset's initial state for each episode.

    Mirrors ``evaluate_tracking_from_dataset`` lines 478-598:
      - load dataset chunks for the selected episodes
      - reset envs with seeds + variations + options
      - apply optional callables (e.g. ``_set_state``, ``_set_goal_state``)
      - restore MuJoCo state via ``mj_setState(mjSTATE_INTEGRATION)`` when
        the dataset has ``mj_state`` column; piecewise ctrl/time/qpos/qvel
        fallback otherwise

    Returns ``(init_step, goal_step)`` — both are dicts of broadcasted
    per-env arrays ready to be merged into ``world.infos`` per step.
    """
    ep_idx_arr = np.array(episode_indices)
    start_steps_arr = np.array(start_steps)
    goal_offset_arr = np.array(goal_offset_steps)
    end_steps = start_steps_arr + goal_offset_arr
    n_envs = world.num_envs
    columns = dataset.column_names
    callables = callables or []

    data = dataset.load_chunk(ep_idx_arr, start_steps_arr, end_steps)

    init_step_per_env: dict = defaultdict(list)
    goal_step_per_env: dict = defaultdict(list)

    _per_episode_cols = {
        "ep_physical_success", "ep_success", "valid_len", "ep_len", "ep_offset",
    }
    for ep in data:
        for col in columns:
            if col in _per_episode_cols or col.startswith("goal"):
                continue
            if col.startswith("pixels"):
                ep[col] = ep[col].permute(0, 2, 3, 1)
            if not isinstance(ep[col], (torch.Tensor, np.ndarray)):
                continue
            if len(ep[col]) == 0:
                continue
            init_data = ep[col][0]
            goal_data = ep[col][-1]
            if not isinstance(init_data, (np.ndarray, torch.Tensor)):
                continue
            init_data = init_data.numpy() if isinstance(init_data, torch.Tensor) else init_data
            goal_data = goal_data.numpy() if isinstance(goal_data, torch.Tensor) else goal_data
            init_step_per_env[col].append(init_data)
            goal_step_per_env[col].append(goal_data)

    init_step = {k: np.stack(v) for k, v in deepcopy(init_step_per_env).items()}
    goal_step = {}
    for key, value in goal_step_per_env.items():
        gk = "goal" if key == "pixels" else f"goal_{key}"
        goal_step[gk] = np.stack(value)

    seeds = init_step.get("seed")
    vkey = "variation."
    variations_dict = {
        k.removeprefix(vkey): v for k, v in init_step.items() if k.startswith(vkey)
    }
    options = [{} for _ in range(n_envs)]
    if variations_dict:
        for i in range(n_envs):
            options[i]["variation"] = list(variations_dict.keys())
            options[i]["variation_values"] = {k: v[i] for k, v in variations_dict.items()}

    init_step.update(deepcopy(goal_step))
    # [Ablation C] optional init-state perturbation: mutate init_step columns
    # (e.g. 'state') BEFORE reset+callables, so _set_state receives the
    # perturbed value and the first rendered observation already reflects it
    # (post-reset mutation would leave stale pixels in world.infos until the
    # first step — the policy's first action would see the UN-perturbed obs).
    if init_perturb_fn is not None:
        init_step = init_perturb_fn(init_step)
    world.reset(seed=seeds, options=options)

    # Apply callables + MuJoCo state restoration.
    import mujoco as _mj
    for i, env in enumerate(world.envs.unwrapped.envs):
        env_unwrapped = env.unwrapped
        for spec in callables:
            method_name = spec["method"]
            if not hasattr(env_unwrapped, method_name):
                continue
            method = getattr(env_unwrapped, method_name)
            args = spec.get("args", spec)
            prepared_args = {}
            for args_name, args_data in args.items():
                value = args_data.get("value", None)
                if args_data.get("in_dataset", True):
                    if value not in init_step:
                        continue
                    prepared_args[args_name] = deepcopy(init_step[value][i])
                else:
                    prepared_args[args_name] = args_data.get("value")
            method(**prepared_args)

        if hasattr(env_unwrapped, "_data"):
            try:
                if "mj_state" in init_step:
                    state = np.asarray(init_step["mj_state"][i], dtype=np.float64)
                    _mj.mj_setState(
                        env_unwrapped._model, env_unwrapped._data, state,
                        _mj.mjtState.mjSTATE_INTEGRATION,
                    )
                else:
                    if "control" in init_step:
                        env_unwrapped._data.ctrl[:] = init_step["control"][i]
                    if "time" in init_step:
                        t0 = init_step["time"][i]
                        env_unwrapped._data.time = float(
                            t0[0] if hasattr(t0, "__len__") else t0
                        )
                    if "prev_qpos" in init_step:
                        env_unwrapped._prev_qpos = init_step["prev_qpos"][i].copy()
                    if "prev_qvel" in init_step:
                        env_unwrapped._prev_qvel = init_step["prev_qvel"][i].copy()
                if hasattr(env_unwrapped, "_model"):
                    _mj.mj_forward(env_unwrapped._model, env_unwrapped._data)
            except Exception as exc:
                print(f"[WARN] env {i} full-state restore partial: {exc}")

    # Verify the restored env really is at the demo frame (opt-in).
    # The dataset columns ARE ``compute_ob_info()`` keys with '/' -> '_', so
    # calling it on the restored env and diffing against init_step is a
    # complete, task-agnostic check rather than a proxy. Load-bearing for
    # mid-episode segment starts, where a silently stale component (button
    # latch, velocity, actuator target) would invalidate the experiment.
    if verify_init:
        worst: dict = {}
        n_checked = 0
        for i, env in enumerate(world.envs.unwrapped.envs):
            u = env.unwrapped
            if not hasattr(u, "compute_ob_info"):
                continue
            try:
                live = u.compute_ob_info()
            except Exception as exc:  # noqa: BLE001
                print(f"[verify_init] env {i}: compute_ob_info failed: {exc}")
                continue
            n_checked += 1
            for k, v in live.items():
                col = k.replace("/", "_")
                if col not in init_step:
                    continue
                try:
                    a = np.asarray(v, dtype=np.float64).ravel()
                    b = np.asarray(init_step[col][i], dtype=np.float64).ravel()
                except (TypeError, ValueError):
                    continue  # non-numeric column (e.g. target_task string)
                if a.shape != b.shape or a.size == 0:
                    continue
                d = float(np.max(np.abs(a - b)))
                if col not in worst or d > worst[col][0]:
                    worst[col] = (d, i)
        if n_checked:
            bad = {k: v for k, v in worst.items() if v[0] > 1e-6}
            print(f"[verify_init] checked {n_checked} envs, {len(worst)} "
                  f"comparable columns, {len(bad)} deviating (>1e-6)")
            for col, (d, i) in sorted(worst.items(), key=lambda kv: -kv[1][0])[:15]:
                flag = "  <-- MISMATCH" if d > 1e-6 else ""
                print(f"[verify_init]   {col:34s} max|live-demo| = {d:.3e} "
                      f"(env {i}){flag}")

    # Broadcast init/goal into the world.infos pixel shape.
    shape_prefix = world.infos["pixels"].shape[:2]
    init_step = {
        k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:])
        for k, v in init_step.items()
    }
    goal_step = {
        k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:])
        for k, v in goal_step.items()
    }
    world.infos.update(deepcopy(init_step))
    world.infos.update(deepcopy(goal_step))

    return init_step, goal_step


# ---------------------------------------------------------------------------
# Phase 3: per-step behavioral-state extraction (task-specific)
# ---------------------------------------------------------------------------

def extract_cube_behavioral(
    world: Any,
    n_envs: int,
    goal_pos_per_env: np.ndarray,
    table_z_per_env: np.ndarray,
) -> dict:
    """Populate ``info['behavioral']`` for cube-task envs.

    Mirrors ``evaluate_tracking_from_dataset`` lines 743-775 behavioral
    capture. Reads cube_pos / effector_pos / gripper_opening directly from
    ``world.envs.unwrapped._data`` (MuJoCo). Returns a dict whose
    ``'behavioral'`` key holds per-env stacked arrays compatible with
    ``CubeGTRMetrics.update``.
    """
    cube_pos_arr = np.zeros((n_envs, 3), dtype=np.float64)
    eff_pos_arr = np.zeros((n_envs, 3), dtype=np.float64)
    grip_arr = np.zeros(n_envs, dtype=np.float64)
    terminated_arr = np.zeros(n_envs, dtype=bool)

    for env_i in range(n_envs):
        try:
            u = world.envs.unwrapped.envs[env_i].unwrapped
            cube_pos_arr[env_i] = u._data.joint("object_joint_0").qpos[:3].copy()
            eff_pos_arr[env_i] = u._data.site_xpos[u._pinch_site_id].copy()
            grip_arr[env_i] = float(u._data.qpos[u._gripper_opening_joint_id]) / 0.8
        except Exception:
            pass
    if world.terminateds is not None:
        terminated_arr = np.asarray(world.terminateds, dtype=bool)

    return {
        "behavioral": {
            "cube_pos": cube_pos_arr,
            "effector_pos": eff_pos_arr,
            "gripper_opening": grip_arr,
            "terminated": terminated_arr,
            "goal_pos": np.asarray(goal_pos_per_env, dtype=np.float64),
            "table_z": np.asarray(table_z_per_env, dtype=np.float64),
        }
    }


def extract_scene_behavioral(world: Any, n_envs: int) -> dict:
    """Populate ``info['behavioral']`` for scene-task envs.

    Reads the five physical quantities that define the scene subtask sequence
    straight off MuJoCo, mirroring the accessors in the demo collector's
    ``check_physical_success`` and in ``SceneEnv._is_in_drawer``:

        drawer_slide qpos              -> stages A / C
        object_joint_0 qpos[:3]        -> stage B (with the handle y below)
        site_xpos[_drawer_site_id][1]  -> stage B box is drawer-relative;
                                          _drawer_site_id is
                                          site('drawer_handle_center').id
        _cur_button_states[1]          -> stage D (latched, NOT in qpos)
        window_slide qpos              -> stage E
        _cur_button_states[0]          -> NOT a stage; it is latched to 1 at
                                          collection reset and never toggled by
                                          the expert. Pressing it drives the
                                          drawer damping to 1e6 (scene_env.py
                                          :460-462) i.e. LOCKS the drawer,
                                          undoing stages A/C. Recorded so the
                                          metric can flag wrong-button presses.

    ``_cur_button_states`` is a plain Python list (not an ndarray) after the
    eval ``set_state`` callable rebuilds it from the dataset columns
    (scene_env.py:229-231), so it is coerced with ``np.asarray`` here rather
    than indexed as an array.
    """
    drawer_qpos = np.zeros(n_envs, dtype=np.float64)
    window_qpos = np.zeros(n_envs, dtype=np.float64)
    block_pos = np.zeros((n_envs, 3), dtype=np.float64)
    handle_y = np.zeros(n_envs, dtype=np.float64)
    button0 = np.ones(n_envs, dtype=np.int64)
    button1 = np.zeros(n_envs, dtype=np.int64)
    eff_pos = np.zeros((n_envs, 3), dtype=np.float64)
    grip = np.zeros(n_envs, dtype=np.float64)
    terminated_arr = np.zeros(n_envs, dtype=bool)

    for env_i in range(n_envs):
        try:
            u = world.envs.unwrapped.envs[env_i].unwrapped
            drawer_qpos[env_i] = float(u._data.joint("drawer_slide").qpos[0])
            window_qpos[env_i] = float(u._data.joint("window_slide").qpos[0])
            block_pos[env_i] = u._data.joint("object_joint_0").qpos[:3].copy()
            handle_y[env_i] = float(u._data.site_xpos[u._drawer_site_id][1])
            _bs = np.asarray(u._cur_button_states).ravel()
            button0[env_i] = int(_bs[0])
            button1[env_i] = int(_bs[1])
            eff_pos[env_i] = u._data.site_xpos[u._pinch_site_id].copy()
            grip[env_i] = float(u._data.qpos[u._gripper_opening_joint_id]) / 0.8
        except Exception:
            pass
    if world.terminateds is not None:
        terminated_arr = np.asarray(world.terminateds, dtype=bool)

    return {
        "behavioral": {
            "drawer_qpos": drawer_qpos,
            "window_qpos": window_qpos,
            "block_pos": block_pos,
            "drawer_handle_y": handle_y,
            "button0_state": button0,
            "button1_state": button1,
            "effector_pos": eff_pos,
            "gripper_opening": grip,
            "terminated": terminated_arr,
        }
    }


def initial_table_z(world: Any, n_envs: int) -> np.ndarray:
    """Read initial cube z (used as table_z proxy for GTR lift threshold)."""
    table_z = np.zeros(n_envs, dtype=np.float64)
    for env_i in range(n_envs):
        try:
            u = world.envs.unwrapped.envs[env_i].unwrapped
            table_z[env_i] = float(u._data.joint("object_joint_0").qpos[2])
        except Exception:
            table_z[env_i] = 0.025
    return table_z


def patch_pusht_block_only_success(
    world: Any,
    pos_dims: tuple[int, ...] = (2, 3),
    angle_dim: int = 4,
    pos_threshold: float = 20.0,
    angle_threshold: float = np.pi / 9,
) -> None:
    """Monkey-patch each sub-env's ``eval_state`` to use BLOCK-only success.

    swm/PushT-v1's stock ``eval_state`` computes pos_diff as a 4-D L2 over
    ``[agent_x, agent_y, block_x, block_y]`` which is unsatisfiable for any
    pushing policy — the expert demos themselves don't satisfy it (0/47 in
    ``expert_demo_025_success``). This helper replaces ``eval_state`` with a
    block-only version (``pos_dims=[2, 3]``), matching the canonical DP-paper
    ``gym_pusht`` "block coverage > 95%" semantic and the 260509 baseline.

    After patching, ``world.terminateds`` (and downstream
    ``info["success"]`` via ``extract_pusht_coverage``) reflects "block at
    goal AND well-rotated", not "block AND agent at demo's final pose".

    Port of ``eval_full.py:178-206`` (block-only patch block).
    """
    import types
    block_dims = list(pos_dims)

    def _block_only_eval_state(self, goal_state, cur_state):
        pos_diff = float(np.linalg.norm(goal_state[block_dims] - cur_state[block_dims]))
        ad = abs(float(goal_state[angle_dim]) - float(cur_state[angle_dim]))
        ad = min(ad, 2 * np.pi - ad)
        success = pos_diff < pos_threshold and ad < angle_threshold
        state_dist = float(np.linalg.norm(goal_state - cur_state))
        return success, state_dist

    for env in world.envs.unwrapped.envs:
        env.unwrapped.eval_state = types.MethodType(_block_only_eval_state, env.unwrapped)


# Canonical PushT goal: block at (256, 256), angle π/4, agent ignored.
# Matches the fixed-goal protocol used by collect_slow_expert_pusht.py and
# the 260509 eval_full.py baseline.
PUSHT_FIXED_GOAL = np.array(
    [256.0, 256.0, 256.0, 256.0, np.pi / 4, 0.0, 0.0], dtype=np.float64
)


def force_pusht_fixed_goal(world: Any) -> None:
    """Force every PushT sub-env to use ``PUSHT_FIXED_GOAL`` after every reset.

    Mirrors the env.reset monkey-patch in ``PushTDPRepoPolicy.reset`` so cells
    that do NOT use that policy (gt_mpcc, sail with custom inners, etc.) still
    evaluate against the canonical fixed goal.

    Why this is necessary: ``reset_world_to_dataset`` filters columns starting
    with "goal" out of ``init_step_per_env``, so a ``_set_goal_state`` callable
    with ``value: goal_state`` receives a SYNTHESIZED value (the chunk's last
    state) — NOT the real dataset goal. For random-start chunks this leaks
    block-near-goal init pairs and produces spurious step-0 successes.

    Idempotent — uses ``env._pusht_fixed_goal_patched`` flag.
    """
    for env in world.envs.unwrapped.envs:
        if not getattr(env, "_pusht_fixed_goal_patched", False):
            _orig_reset = env.reset

            def _patched(*args, _orig=_orig_reset, _env=env, **kwargs):
                obs, info = _orig(*args, **kwargs)
                _env.unwrapped._set_goal_state(PUSHT_FIXED_GOAL.copy())
                return obs, info

            env.reset = _patched
            env._pusht_fixed_goal_patched = True
        env.unwrapped._set_goal_state(PUSHT_FIXED_GOAL.copy())


def extract_pusht_coverage(world: Any, n_envs: int) -> dict:
    """Populate info['coverage'] + info['success'] from PushT env state.

    The swm PushT env populates `info['coverage']` = IoU(block, goal) ∈ [0,1]
    in `_get_info()`. PushTCoverageMetrics expects per-env coverage under
    `info['coverage']`. `info['reward']` from the env is `-distance` and is
    NOT coverage — do not use it.

    `success` is sourced from `world.terminateds`, which encodes the env's
    `eval_state()` rule (pos<20 AND angle<π/9 against `goal_state`). When
    a downstream policy applies the canonical `_set_goal_state(PUSHT_FIXED_GOAL)`
    fix (see `policy.pusht_dp_repo`), this matches DP-paper semantics.
    """
    out: dict = {}
    if "coverage" in world.infos:
        cov = np.asarray(world.infos["coverage"])
        if cov.ndim > 1:  # (n_envs, history) — take latest
            cov = cov[..., -1]
        out["coverage"] = cov
    if world.terminateds is not None:
        out["success"] = np.asarray(world.terminateds, dtype=bool)
    return out


# ---------------------------------------------------------------------------
# Granular (PyFleX) helpers — mirror the pusht patterns above.
# DeformableEnv has a single C-global state so we always index env 0 here.
# ---------------------------------------------------------------------------

def extract_granular_chamfer(world: Any, n_envs: int) -> dict:
    """Populate info['chamfer'] + info['success'] from DeformableEnv state.

    DeformableEnv exposes ``chamfer_to_goal() -> {'chamfer_distance': float}``
    against the goal_state set at reset. We surface that as a per-env array
    so ``GranularChamferMetrics.update`` can read it via ``_per_env``.

    n_envs must be 1 here — PyFleX has process-global C state so only one
    DeformableEnv can live per process. We still emit a (n_envs,) shaped
    array for schema consistency with pusht/cube.
    """
    out: dict = {}
    env0 = world.envs.unwrapped.envs[0].unwrapped
    m = env0.chamfer_to_goal()
    cd = float(m["chamfer_distance"])
    out["chamfer"] = np.full(n_envs, cd, dtype=np.float64)
    out["success"] = np.full(n_envs, cd < 0.03, dtype=bool)
    return out


def force_granular_goal(world: Any) -> None:
    """Force every granular sub-env to keep the dataset goal across resets.

    Why this is necessary: ``reset_world_to_dataset`` filters columns starting
    with ``"goal"`` out of ``init_step_per_env`` (see line 99 above), so a
    naive reset would clobber the dataset's saved goal_state with whatever
    DeformableEnv samples internally. We monkey-patch ``env.reset`` so each
    reset() call rebinds the env's internal ``_goal_state`` back to the
    value captured at the FIRST reset (which IS the dataset goal, because
    reset_world_to_dataset passes ``options={'goal_state': dataset_goal}`` to
    that first call).

    Idempotent — uses ``env._granular_goal_patched`` flag.
    """
    for env in world.envs.unwrapped.envs:
        if not getattr(env, "_granular_goal_patched", False):
            _orig_reset = env.reset

            def _patched(*args, _orig=_orig_reset, _env=env, **kwargs):
                # First call: capture the dataset goal that swm passed in.
                opts = kwargs.get("options") or {}
                injected_goal = opts.get("goal_state", None)
                obs, info = _orig(*args, **kwargs)
                # Remember the dataset goal for any FUTURE resets that lack it
                # (e.g. gym TimeLimit auto-reset paths). Idempotent capture.
                if injected_goal is not None and getattr(
                    _env, "_granular_dataset_goal", None
                ) is None:
                    _env._granular_dataset_goal = np.asarray(
                        injected_goal, dtype=np.float32
                    ).reshape(-1, 4)
                stored = getattr(_env, "_granular_dataset_goal", None)
                if stored is not None:
                    _env.unwrapped._goal_state = stored.copy()
                    # Refresh the rendered goal pixels to match.
                    if hasattr(_env.unwrapped, "_render_goal"):
                        _env.unwrapped._render_goal()
                return obs, info

            env.reset = _patched
            env._granular_goal_patched = True
