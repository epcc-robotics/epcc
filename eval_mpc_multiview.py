from __future__ import annotations  # py3.9 compat: defer PEP 604 union evaluation
"""Multi-view MPC evaluation: time-indexed trajectory tracking in latent space
using BOTH agentview pixels and wrist_pixels.

Identical to eval_mpc.py except:
  - loads wrist_pixels alongside pixels
  - uses MultiviewMPCSolver (subclass of latent_modules.mpc.MPCSolver) so reference
    trajectories are encoded through MultiViewJEPA (both camera streams)
  - renders wrist_pixels per env step and injects into world.infos
    (the env itself only emits agentview pixels; we fake the wrist cam)
  - patches stable_worldmodel.policy._prepare_info to handle wrist_pixels
    layout (HWC/CHW asymmetry)

The eval loop, success metric, reset_steps / advance_steps logic is byte-for-byte
the single-view version. Only input shape (dual-camera) differs.

Usage:
    python eval_mpc_multiview.py --config-name multiview_cube \
        solver=mpc_multiview policy=mv_cube_wm_training \
        eval.dataset_name="Multiview WM/cube_multiview_diffusion"
"""

import os
import sys

os.environ["MUJOCO_GL"] = "egl"


def _silence_egl_unraisable(unraisable):
    """Drop harmless EGL/OpenGL destructor spam at Python interpreter shutdown."""
    exc_type = unraisable.exc_type
    exc_module = getattr(exc_type, "__module__", "") or ""
    if "OpenGL" in exc_module or "EGL" in exc_type.__name__:
        return
    sys.__unraisablehook__(unraisable)


sys.unraisablehook = _silence_egl_unraisable

import json
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
import imageio
import mujoco
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

from latent_modules.mpc import MPCSolver, MPCCostModelDebug


# ---------------------------------------------------------------------------
# Multi-view plumbing (layered on top of the single-view eval_mpc.py)
# ---------------------------------------------------------------------------

def _patch_swm_policy_for_wrist_pixels():
    """stable_worldmodel.policy.WorldModelPolicy._prepare_info only auto-permutes
    keys matching `startswith('pixels')` or `startswith('goal')` (HWC→CHW before
    torchvision transforms). We also need 'wrist_pixels' and 'goal_wrist_pixels'
    handled symmetrically."""
    _orig = swm.policy.WorldModelPolicy._prepare_info

    def _to_chw(v):
        if v.ndim >= 3 and v.shape[-1] in (1, 3, 4) and v.shape[-3] not in (1, 3, 4):
            perm = list(range(v.ndim - 3)) + [v.ndim - 1, v.ndim - 3, v.ndim - 2]
            return np.transpose(v, perm) if isinstance(v, np.ndarray) else v.permute(*perm)
        return v

    def _to_hwc(v):
        if v.ndim >= 3 and v.shape[-3] in (1, 3, 4) and v.shape[-1] not in (1, 3, 4):
            perm = list(range(v.ndim - 3)) + [v.ndim - 2, v.ndim - 1, v.ndim - 3]
            return np.transpose(v, perm) if isinstance(v, np.ndarray) else v.permute(*perm)
        return v

    def _prepare_info_mv(self, info_dict):
        if "wrist_pixels" in info_dict:
            info_dict["wrist_pixels"] = _to_chw(info_dict["wrist_pixels"])
        if "goal_wrist_pixels" in info_dict:
            info_dict["goal_wrist_pixels"] = _to_hwc(info_dict["goal_wrist_pixels"])
        return _orig(self, info_dict)

    swm.policy.WorldModelPolicy._prepare_info = _prepare_info_mv


_patch_swm_policy_for_wrist_pixels()


class WristCameraRenderer:
    """First-person eye-in-hand via direct mjvScene.camera override.
    Copied from scripts/collect_multiview_WM_scene.py."""

    def __init__(self, model, height=224, width=224):
        self.model = model
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultFreeCamera(model, self.cam)

        self._site_id = None
        for i in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name and "attachment_site" in name:
                self._site_id = i
                break
        if self._site_id is None:
            print("[eval_mpc_multiview] attachment_site not found — wrist cam disabled")

    def render(self, data):
        if self._site_id is None:
            return np.zeros((self.renderer.height, self.renderer.width, 3), dtype=np.uint8)

        ee_pos = data.site_xpos[self._site_id].copy()
        ee_mat = data.site_xmat[self._site_id].reshape(3, 3).copy()
        forward_raw = ee_mat[:, 2]
        up_raw = ee_mat[:, 1]

        tilt = 0.175
        forward = forward_raw - up_raw * tilt
        forward = forward / np.linalg.norm(forward)
        up = up_raw - np.dot(up_raw, forward) * forward
        up = up / np.linalg.norm(up)
        cam_pos = ee_pos + up_raw * 0.06 - forward_raw * 0.015

        self.renderer.update_scene(data, camera=self.cam, scene_option=self.scene_option)
        scene = self.renderer.scene
        for i in range(2):
            scene.camera[i].pos[:] = cam_pos.astype(np.float32)
            scene.camera[i].forward[:] = forward.astype(np.float32)
            scene.camera[i].up[:] = up.astype(np.float32)

        return self.renderer.render().copy()


_wrist_renderers: list = []


def _attach_wrist_renderers(world, img_size: int = 224) -> None:
    """Create one WristCameraRenderer per env and monkey-patch World.step to
    render + inject `wrist_pixels` into world.infos after every env step.
    The env itself only produces agentview 'pixels'; this fills in the wrist
    stream the multiview model expects."""
    global _wrist_renderers
    if _wrist_renderers:
        return

    for env in world.envs.unwrapped.envs:
        _wrist_renderers.append(
            WristCameraRenderer(env.unwrapped._model, height=img_size, width=img_size)
        )

    _orig_step = swm.World.step

    def _step_with_wrist(self):
        _orig_step(self)
        envs = self.envs.unwrapped.envs
        wrist = np.stack(
            [r.render(env.unwrapped._data) for r, env in zip(_wrist_renderers, envs)]
        )  # (N, H, W, C)
        if "pixels" in self.infos and self.infos["pixels"].ndim == wrist.ndim + 1:
            wrist = np.broadcast_to(
                wrist[:, None],
                (wrist.shape[0], self.infos["pixels"].shape[1], *wrist.shape[1:]),
            ).copy()
        self.infos["wrist_pixels"] = wrist

    swm.World.step = _step_with_wrist


_proprio_alias_installed = False


def _attach_proprio_alias(world, proprio_keys: tuple[str, ...]) -> None:
    """Monkey-patch World.step so each runtime info_dict exposes proprio
    columns under the underscore-separated names that PerViewJEPA.encode
    expects (e.g. "proprio_effector_pos") in addition to the slash-separated
    names the env actually emits ("proprio/effector_pos"). Also run once
    immediately to populate keys before the first step is taken.
    Idempotent — safe to call multiple times."""
    global _proprio_alias_installed
    if _proprio_alias_installed:
        return
    _proprio_alias_installed = True

    def _rename(infos, protect_existing=False):
        for k in list(infos.keys()):
            if "/" in k and k.startswith("proprio/"):
                new_k = k.replace("/", "_", 1)
                # 🔴 protect_existing guards exactly one instant: between
                # world.reset() and the first envs.step(). The slash keys are
                # captured once by envs.reset() and are NOT refreshed when
                # evaluate_tracking_from_dataset then applies set_state /
                # mj_setState to move the arm onto the demo pose, so at t=0
                # they still hold the random reset pose. The underscore keys,
                # by contrast, were just injected from the demo h5
                # (`world.infos.update(deepcopy(init_step))`) and are correct.
                # Overwriting them here paired demo pixels with a pose the arm
                # is not in; measured offset on cube was effector_pos 0.2209 m
                # = 1.4-3.1 sigma of the training scaler. In DTG-MPCC the first
                # embedding is replicated across the whole history ring, so that
                # bad state steered the entire first plan. From step 1 on the
                # slash keys are fresh and must win, which is what every later
                # call does. Only AWP-style models read proprio at all.
                if protect_existing and new_k in infos:
                    continue
                infos[new_k] = infos[k]

    _orig = swm.World.step
    _first_step = [True]

    def _step_with_proprio_alias(self):
        # Rename BEFORE step so policy.get_action (called inside World.step
        # via envs.step loop) sees underscore keys the encoder expects.
        # The prior step's renamed keys already persist; this handles fresh
        # slash-keyed infos produced by envs.step at the PREVIOUS step.
        if _first_step[0]:
            # Report what the guard is worth before it takes effect.
            for k in list(self.infos.keys()):
                if not k.startswith("proprio/"):
                    continue
                u = k.replace("/", "_", 1)
                if u not in self.infos:
                    continue
                try:
                    a = np.asarray(self.infos[u], dtype=np.float64)
                    b = np.asarray(self.infos[k], dtype=np.float64)
                    d = float(np.abs(a[:, -1] - b.reshape(a.shape)[:, -1]).max())
                    print(f"[proprio t=0] {u}: kept demo value; "
                          f"stale reset pose differed by max {d:.4f}")
                except Exception as exc:
                    print(f"[proprio t=0] {u}: kept demo value (delta not computed: {exc})")
        _rename(self.infos, protect_existing=_first_step[0])
        if _first_step[0]:
            _first_step[0] = False
        _orig(self)
        # Post-step rename so subsequent calls that read world.infos directly
        # (e.g. per-env behavioral state capture) also see underscore keys.
        _rename(self.infos)

    swm.World.step = _step_with_proprio_alias
    # Populate immediately since evaluate_tracking_from_dataset calls
    # world.reset AFTER this hook installs, so any post-reset get_action
    # would miss the rename without this pre-fill of current infos.
    _rename(world.infos)
    print(f"[proprio alias] attached — "
          f"runtime keys: {[k.replace('/', '_', 1) for k in proprio_keys]}")


def _ensure_hwc_numpy_for_transform(arr):
    """torchvision v2 ToImage() only auto-permutes HWC→CHW when the input is a
    numpy array — for a torch tensor it's just a type-wrap. Return numpy HWC."""
    if hasattr(arr, "cpu") and hasattr(arr, "numpy"):
        arr = arr.cpu().numpy()
    if arr.ndim == 4 and arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (0, 2, 3, 1))
    return arr


class MultiviewMPCSolver(MPCSolver):
    """MPCSolver that encodes reference trajectories through a PerViewJEPA.
    Dispatches on `model.enabled_views` — loads pixels + wrist_pixels (AW),
    pixels + proprio columns (AP), or all three. Dim-agnostic downstream."""

    def _encode_episode(
        self,
        dataset_name: str,
        episode_idx: int,
        start_step: int,
        end_step: int,
        chunk_size: int = 64,
        cache_dir: str | None = None,
    ) -> torch.Tensor:
        enabled_views = getattr(self._model, "enabled_views", ("agent", "wrist"))
        proprio_keys = list(getattr(self._model, "proprio_keys", ()) or ())

        keys_to_load = ["pixels"]
        if "wrist" in enabled_views:
            keys_to_load.append("wrist_pixels")
        if "proprio" in enabled_views:
            keys_to_load.extend(proprio_keys)

        kwargs: dict[str, Any] = {
            "name": dataset_name,
            "frameskip": 1,
            "keys_to_load": keys_to_load,
        }
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        episode = swm.data.HDF5Dataset(**kwargs).load_episode(episode_idx)

        transform = self._transform
        pix_raw = _ensure_hwc_numpy_for_transform(episode["pixels"][start_step:end_step])
        pix = torch.stack([transform(p) for p in pix_raw])

        wrist = None
        if "wrist" in enabled_views:
            wrist_raw = _ensure_hwc_numpy_for_transform(episode["wrist_pixels"][start_step:end_step])
            wrist = torch.stack([transform(w) for w in wrist_raw])

        proprio_tensors: dict[str, torch.Tensor] = {}
        if "proprio" in enabled_views:
            # Scalers set by run() via solver._scalers — required for AP because
            # PerViewJEPA training normalized proprio columns
            # (train_perview_WM_cube.py:344 get_column_normalizer). Raw-pass
            # would feed OOD values to the ProprioEncoder MLP.
            scalers = getattr(self, "_scalers", {}) or {}
            for k in proprio_keys:
                arr = episode[k][start_step:end_step]
                if hasattr(arr, "cpu"):
                    arr = arr.cpu().numpy()
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[:, None]
                if k in scalers:
                    arr = scalers[k].transform(arr).astype(np.float32)
                proprio_tensors[k] = torch.from_numpy(arr)

        is_prejepa = hasattr(self._model, "extra_encoders") and self._model.extra_encoders
        encode_kwargs: dict[str, Any] = {}
        if is_prejepa:
            encode_kwargs["emb_keys"] = []

        embs = []
        with torch.no_grad():
            for i in range(0, len(pix), chunk_size):
                info = {
                    "pixels": pix[i:i + chunk_size].unsqueeze(0).to(self._device),
                }
                if wrist is not None:
                    info["wrist_pixels"] = wrist[i:i + chunk_size].unsqueeze(0).to(self._device)
                for k, t in proprio_tensors.items():
                    info[k] = t[i:i + chunk_size].unsqueeze(0).to(self._device)
                self._model.encode(info, **encode_kwargs)
                emb = info["emb"].squeeze(0)
                embs.append(emb.cpu())
        return torch.cat(embs).to(self._device)


class MultiviewMPCSolverDebug(MultiviewMPCSolver):
    """Debug variant: plugs ``MPCCostModelDebug`` for per-env breakdown
    collection. Structurally mirrors :class:`latent_modules.mpc.MPCSolverDebug` but
    inherits the multiview ``_encode_episode`` from the non-debug MV solver.
    Select via Hydra: ``solver=mpc_multiview_debug``.
    """

    COST_MODEL_CLS = MPCCostModelDebug

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        # Fresh per-env dict every solve so stale breakdowns don't leak.
        self.cost_model._env_breakdowns.clear()
        return super().solve(info_dict, init_action=init_action)


class MultiviewWorldModelPolicy(swm.policy.WorldModelPolicy):
    """MV-aware ``WorldModelPolicy``: pre-fills the past-action buffer with
    zeros at ``set_env`` time so the upstream warm-up fallback never fires.

    Upstream's warm-up fallback (commit 47e7a3d) slices only ``info['pixels']``
    to ``size=1`` on the time dim while other view tensors (``wrist_pixels``,
    ``proprio_*``) stay at ``history_size`` — ``PerViewJEPA.encode`` then hits
    a ``torch.cat`` size mismatch. Pre-filling the deque with zeros makes
    ``have_full_history=True`` from step 1, so the branch that slices pixels
    is never taken. Zeros are treated by the model as "no prior motion",
    which matches our pre-47e7a3d behavior.
    """

    def set_env(self, env):
        super().set_env(env)
        if self._past_action_buffer is not None and self._past_action_cap > 0:
            n_envs = getattr(env, "num_envs", 1)
            raw_A = int(np.prod(env.action_space.shape[1:]))
            zeros = torch.zeros(n_envs, raw_A)
            for _ in range(self._past_action_cap):
                self._past_action_buffer.append(zeros.clone())


def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def get_episodes_length(dataset, episodes):
    # Preferred path: derive per-episode length from per-frame step_idx.
    if "step_idx" in dataset.column_names:
        col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        episode_idx = dataset.get_col_data(col_name)
        step_idx = dataset.get_col_data("step_idx")
        lengths = []
        for ep_id in episodes:
            lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
        return np.array(lengths)
    # Fallback for datasets without step_idx (e.g. c-sweep ablation /
    # expert_100): ep_len is a per-episode length array indexed by episode id.
    ep_len = np.asarray(dataset.get_col_data("ep_len"))
    return np.array([int(ep_len[int(ep_id)]) for ep_id in episodes])


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


# ---------------------------------------------------------------------------
# MPC evaluation loop (mirrors eval_mpcc.mpcc_evaluate with step tracking)
# ---------------------------------------------------------------------------

def _compute_episode_metrics(
    states: list,
    goal_pos,
    table_z: float,
    lift_thresh: float = 0.02,
    min_lift_steps: int = 3,
    hold_dist_thresh: float = 0.05,
    drop_jump_thresh: float = 0.02,
    goal_dist_thresh: float = 0.07,
) -> dict:
    """Behavioral success metrics for one episode. Mirrors eval_mpc._compute_episode_metrics."""
    from itertools import groupby
    if not states:
        return {'grasped': False, 'no_drop': False, 'final_dist': float('inf'),
                'reached_goal': False, 'success_full': False,
                'first_grasp_step': -1, 'longest_grasp_run': 0, 'drop_step': -1}
    cube_pos = np.array([s['cube_pos'] for s in states])
    cube_z = cube_pos[:, 2]
    has_eff = 'effector_pos' in states[0]
    if has_eff:
        eff_pos = np.array([s['effector_pos'] for s in states])
        eff_cube_dist = np.linalg.norm(eff_pos - cube_pos, axis=1)
    else:
        eff_cube_dist = np.full(len(states), np.inf)

    lifted = cube_z > (table_z + lift_thresh)
    near = eff_cube_dist < hold_dist_thresh
    grasp = lifted & near
    longest = max((sum(1 for _ in g) for k, g in groupby(grasp.tolist()) if k), default=0)
    grasped = longest >= min_lift_steps
    first_grasp = int(np.argmax(grasp)) if grasp.any() else -1

    no_drop = False
    drop_step = -1
    if grasped and first_grasp >= 0 and has_eff:
        no_drop = True
        ddist = np.abs(np.diff(eff_cube_dist))
        # Elevation guard: cube must be well above table for release to count
        # as drop (not placement). Previous `table_z + lift_thresh/2` = 1cm
        # caused false drops when gripper opened during low-altitude placement.
        elevation_guard = table_z + 0.03   # 3cm above table
        # Placement exception: near-goal release is valid placement, not drop.
        # Skip drop detection when cube is within `placement_proximity` of goal.
        placement_proximity = 0.12   # 12cm — generous radius around goal
        goal_arr = np.asarray(goal_pos)
        for i in range(first_grasp, len(states) - 1):
            cube_goal_dist = np.linalg.norm(cube_pos[i] - goal_arr)
            if cube_goal_dist < placement_proximity:
                # Cube near goal → gripper open = placement, not drop
                continue
            if ddist[i] > drop_jump_thresh and cube_z[i] > elevation_guard:
                no_drop = False
                drop_step = i
                break

    final_cube = cube_pos[-1]
    final_dist = float(np.linalg.norm(final_cube - np.asarray(goal_pos)))
    reached_goal = final_dist < goal_dist_thresh

    return {
        'grasped': bool(grasped),
        'no_drop': bool(no_drop),
        'final_dist': final_dist,
        'reached_goal': bool(reached_goal),
        'success_full': bool(grasped and no_drop and reached_goal),
        'first_grasp_step': first_grasp,
        'longest_grasp_run': int(longest),
        'drop_step': drop_step,
    }


def evaluate_tracking_from_dataset(
    world,
    dataset,
    solver,
    episodes_idx,
    start_steps,
    goal_offset_steps,
    eval_budget,
    callables=None,
    video_path="./",
    save_video=True,
    debug=False,
    model=None,
    logger=None,
):
    """Run MPC evaluation on reference trajectories.

    Same structure as mpcc_evaluate but advances solver step counter each step.

    ``logger``: optional ``latent_modules.eval.RolloutLogger``. When provided, each
    env-step records action / reward / behavioral state / per-spec success
    flags; ``logger.finalize`` is called at end of the loop to write
    rollout.npz + meta.yaml + keyframes.
    """
    ep_idx_arr = np.array(episodes_idx)
    start_steps_arr = np.array(start_steps)
    goal_offset_arr = np.array(goal_offset_steps)
    end_steps = start_steps_arr + goal_offset_arr
    n_envs = world.num_envs
    columns = dataset.column_names

    # -- Load chunks from dataset ----------------------------------------------
    data = dataset.load_chunk(ep_idx_arr, start_steps_arr, end_steps)

    init_step_per_env = defaultdict(list)
    goal_step_per_env = defaultdict(list)

    # Per-episode columns (shape (n_episodes,)) get mis-sliced as per-step by
    # load_chunk — skip them to avoid broadcast errors with large n_eval.
    _per_episode_cols = {"ep_physical_success", "ep_success", "valid_len", "ep_len", "ep_offset"}
    for i, ep in enumerate(data):
        for col in columns:
            if col in _per_episode_cols:
                continue
            if col.startswith("goal"):
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

    # Reset envs
    seeds = init_step.get("seed")
    vkey = "variation."
    variations_dict = {
        k.removeprefix(vkey): v for k, v in init_step.items() if k.startswith(vkey)
    }
    options = [{} for _ in range(n_envs)]
    if len(variations_dict) > 0:
        for i in range(n_envs):
            options[i]["variation"] = list(variations_dict.keys())
            options[i]["variation_values"] = {
                k: v[i] for k, v in variations_dict.items()
            }

    init_step.update(deepcopy(goal_step))
    world.reset(seed=seeds, options=options)

    # -- Apply callables (e.g. _set_state, _set_goal_state) --------------------
    callables = callables or []
    import mujoco as _mj
    _restore_mode = None   # logged once after the env-0 restore path
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

        # Full-state restoration — mirrors eval_mpc.py SV path. Uses
        # mj_setState(mjSTATE_INTEGRATION) fast path when dataset has
        # `mj_state` column (bit-perfect reproduction); falls back to
        # piecewise ctrl/time/prev_qpos/prev_qvel injection otherwise.
        if hasattr(env_unwrapped, "_data"):
            try:
                if "mj_state" in init_step:
                    state = np.asarray(init_step["mj_state"][i], dtype=np.float64)
                    _mj.mj_setState(
                        env_unwrapped._model, env_unwrapped._data, state,
                        _mj.mjtState.mjSTATE_INTEGRATION,
                    )
                    if i == 0:
                        _restore_mode = "mj_state (INTEGRATION) — bit-perfect"
                else:
                    restored = []
                    if "control" in init_step:
                        env_unwrapped._data.ctrl[:] = init_step["control"][i]
                        restored.append("ctrl")
                    if "time" in init_step:
                        t0 = init_step["time"][i]
                        env_unwrapped._data.time = float(
                            t0[0] if hasattr(t0, "__len__") else t0
                        )
                        restored.append("time")
                    if "prev_qpos" in init_step:
                        env_unwrapped._prev_qpos = init_step["prev_qpos"][i].copy()
                        restored.append("prev_qpos")
                    if "prev_qvel" in init_step:
                        env_unwrapped._prev_qvel = init_step["prev_qvel"][i].copy()
                        restored.append("prev_qvel")
                    if i == 0:
                        _restore_mode = (
                            f"piecewise ({', '.join(restored) or 'qpos/qvel only'})"
                        )
                if hasattr(env_unwrapped, "_model"):
                    _mj.mj_forward(env_unwrapped._model, env_unwrapped._data)
            except Exception as exc:
                print(f"[WARN] env {i} full-state restore partial: {exc}")
    if _restore_mode is not None:
        print(f"[full-state restore] {_restore_mode}")

    # -- Optional: verify the restored env state against the demo frame ---------
    # ``+eval.verify_init=true``. The dataset columns ARE ``compute_ob_info()``
    # keys (with '/' -> '_'), so calling it on the restored env and diffing
    # against init_step is a complete, task-agnostic check that the env really
    # is at the demo frame — not a proxy for it. Load-bearing for mid-episode
    # segment starts, where a silently stale component (button latch, velocity,
    # actuator target) would invalidate the whole experiment.
    if bool(cfg.eval.get("verify_init", False)):
        worst: dict[str, tuple[float, int]] = {}
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
            print(f"[verify_init] checked {n_checked} envs, "
                  f"{len(worst)} comparable columns, "
                  f"{len(bad)} deviating (>1e-6)")
            for col, (d, i) in sorted(worst.items(), key=lambda kv: -kv[1][0])[:12]:
                flag = "  <-- MISMATCH" if d > 1e-6 else ""
                print(f"[verify_init]   {col:34s} max|live-demo| = {d:.3e} "
                      f"(env {i}){flag}")

    # -- Broadcast init/goal into info shape ------------------------------------
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

    # -- Tracking arrays -------------------------------------------------------
    target_frames = [
        ep["pixels"].numpy() if isinstance(ep["pixels"], torch.Tensor) else np.array(ep["pixels"])
        for ep in data
    ]  # list of (T_i, H, W, C) — may have different T_i per env
    video_frames = np.empty(
        (n_envs, eval_budget, *world.infos["pixels"].shape[-3:]),
        dtype=np.uint8,
    )
    # Optional wrist-view recording (only when wrist_pixels is in infos).
    # wrist_pixels may be CHW (after _prepare_info) or HWC (raw). Pre-allocate
    # to HWC matching pixels shape; transpose at write time if needed.
    record_wrist = "wrist_pixels" in world.infos
    wrist_frames = (
        np.empty((n_envs, eval_budget, *world.infos["pixels"].shape[-3:]),
                 dtype=np.uint8) if record_wrist else None
    )

    def _to_hwc_uint8(arr):
        """Coerce wrist_pixels[:, -1] (shape can be N,C,H,W or N,H,W,C; uint8 or float) → N,H,W,C uint8."""
        a = np.asarray(arr)
        if a.shape[-3] == 3 and a.shape[-1] != 3:
            a = np.transpose(a, (0, 2, 3, 1))
        if a.dtype != np.uint8:
            if a.max() <= 1.0:
                a = (a * 255).clip(0, 255).astype(np.uint8)
            else:
                a = a.clip(0, 255).astype(np.uint8)
        return a

    episode_successes = np.zeros(n_envs, dtype=bool)
    first_success_step = np.full(n_envs, -1, dtype=np.int64)
    distances = np.full((n_envs, eval_budget), np.nan)
    rewards = np.full((n_envs, eval_budget), np.nan)

    # Per-env behavioral state buffers — captured each env-step for end-of-episode
    # scoring: stable grasp, no-drop transport, final goal dist. Each env's buffer
    # FREEZES once it satisfies G+T+R so a stray arm-bump can't unset success.
    episode_states_per_env = [[] for _ in range(n_envs)]
    freeze_step_per_env = np.full(n_envs, -1, dtype=np.int64)
    goal_pos_per_env = goal_step.get("goal_privileged_block_0_pos")
    if goal_pos_per_env is None:
        goal_pos_per_env = np.zeros((n_envs, 3))
        print("[behavioral] WARNING: goal_privileged_block_0_pos missing — final_dist=0")
    table_z_per_env = np.zeros(n_envs)
    for env_i in range(n_envs):
        try:
            u = world.envs.unwrapped.envs[env_i].unwrapped
            table_z_per_env[env_i] = float(u._data.joint("object_joint_0").qpos[2])
        except Exception:
            table_z_per_env[env_i] = 0.025

    # Reset solver step counters
    if solver is not None:
        solver.reset_steps()

    # -- RolloutLogger action capture ------------------------------------------
    # Wrap policy.get_action so the action that lands in env.step is stashed
    # for per-step logger.log_step. Idempotent — re-running this in the same
    # process is safe because we guard via ``_logger_action_capture_installed``.
    if logger is not None and not getattr(
        world.policy, "_logger_action_capture_installed", False
    ):
        _orig_get_action = world.policy.get_action

        def _capture_get_action(info_dict, **kw):
            out = _orig_get_action(info_dict, **kw)
            try:
                world.policy._last_action_executed = (
                    out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
                )
            except Exception:
                world.policy._last_action_executed = None
            return out

        world.policy.get_action = _capture_get_action
        world.policy._logger_action_capture_installed = True

    # -- Debug logger ----------------------------------------------------------
    if debug and solver is not None:
        from latent_modules.mpc.debug_video import DebugLogger
        debug_logger = DebugLogger(n_envs, eval_budget)
    else:
        debug_logger = None

    # -- Per-env cost breakdowns (from MultiviewMPCSolverDebug) ---------------
    # Populated only when solver=mpc_multiview_debug is selected at the Hydra
    # CLI. With base solver=mpc_multiview, ``_env_breakdowns`` doesn't exist
    # and the step-loop consumer defaults to an empty dict. See
    # config/eval/solver/mpc_multiview_debug.yaml.
    # CEM convergence summary ([env X] cost A → B (×r), N evals) is printed
    # by CEMSolver.solve itself — no external wrapping needed.

    # -- Step loop -------------------------------------------------------------
    for t in range(eval_budget):
        video_frames[:, t] = world.infos["pixels"][:, -1]
        if record_wrist:
            wrist_frames[:, t] = _to_hwc_uint8(world.infos["wrist_pixels"][:, -1])
        world.infos.update(deepcopy(goal_step))
        world.step()

        # Advance solver's reference step counter by action_block
        # (each world.step() executes action_block low-level steps)
        solver.advance_steps()

        # =====================================================================
        # Per-env behavioral state capture for
        # G/T/R 3-criterion success decomposition (Grasped / no-drop Transport
        # / Reached goal). The single-view eval uses only ``world.terminateds`` — too
        # permissive on cube (brief goal proximity flips terminated=True).
        # We record cube_pos / effector_pos / gripper each env-step into
        # ``episode_states_per_env`` and scan at episode-end via
        # ``_compute_episode_metrics`` to compute G/T/R independently.
        #
        # Early freeze: when cube enters 5cm of goal AND G+T+R all satisfy,
        # lock via ``freeze_step_per_env[env_i] = t`` and skip further appends.
        # Prevents arm-bump after success from flipping reached_goal=False
        # at end-of-episode scoring. Overhead per step: negligible.
        # =====================================================================
        # ``per_env_state_for_logger`` collects the SAME per-env state dicts
        # exposed to RolloutLogger.log_step's success specs so naive + gtr
        # share a consistent view of the running history.
        #
        # NOTE on freeze: the eval-loop's freeze (cube reached goal AND GTR
        # satisfied) only stops appends to ``episode_states_per_env`` so that
        # end-of-episode behavioral scoring is locked at the success window.
        # We must STILL read cube_pos / effector_pos / gripper for the logger
        # — otherwise the logger's gtr SuccessSpec KeyErrors on a stripped
        # state dict and the entire per-step trace truncates from there.
        per_env_state_for_logger: list = [{} for _ in range(n_envs)]
        for env_i in range(n_envs):
            try:
                u = world.envs.unwrapped.envs[env_i].unwrapped
                cube_pos = u._data.joint("object_joint_0").qpos[:3].copy()
                eff_pos = u._data.site_xpos[u._pinch_site_id].copy()
                grip = float(u._data.qpos[u._gripper_opening_joint_id]) / 0.8
                per_env_state_for_logger[env_i] = {
                    'cube_pos': cube_pos,
                    'effector_pos': eff_pos,
                    'gripper_opening': grip,
                    'goal_pos': np.asarray(goal_pos_per_env[env_i]),
                    'table_z': float(table_z_per_env[env_i]),
                }
                if freeze_step_per_env[env_i] >= 0:
                    # Frozen — eval's behavioral scoring won't see post-freeze
                    # state, but logger does.
                    continue
                episode_states_per_env[env_i].append({
                    'step': t, 'cube_pos': cube_pos,
                    'effector_pos': eff_pos, 'gripper_opening': grip,
                })
                gp = np.asarray(goal_pos_per_env[env_i])
                if np.linalg.norm(cube_pos - gp) < 0.07:
                    m_now = _compute_episode_metrics(
                        episode_states_per_env[env_i],
                        goal_pos=gp, table_z=table_z_per_env[env_i],
                    )
                    if m_now['success_full']:
                        freeze_step_per_env[env_i] = t
                        print(f"  [behavioral] env {env_i} GTR locked at step {t}")
            except Exception:
                pass

        # RolloutLogger hook — opt-in dump pipeline for the paper plot scripts.
        # Records executed env action, terminated signal, behavioral state
        # (so both naive + gtr SuccessSpec callables run consistently), and
        # reward. Latent / theta left NaN here because they're solver-internal.
        if logger is not None:
            try:
                last_action = getattr(world.policy, "_last_action_executed", None)
                if hasattr(last_action, "cpu"):
                    last_action = last_action.cpu().numpy()
                last_action = (np.asarray(last_action, dtype=np.float32)
                               if last_action is not None else None)
                logger.log_step(
                    t,
                    action=last_action,
                    reward=(world.rewards if world.rewards is not None else None),
                    terminated=np.asarray(world.terminateds, dtype=bool),
                    env_states=per_env_state_for_logger,
                )
            except Exception as exc:
                if t == 0:
                    print(f"[RolloutLogger] WARN: log_step failed: {exc!r}")

        if debug_logger is not None:
            debug_logger.log_step(t, solver)

            # =================================================================
            # Multi-batch CEM debug_video fix.
            # ``debug_logger.log_step`` reads ``cost_model._last_breakdown``,
            # which CEMSolver overwrites once per batch. With batch_size=25 ×
            # 2 batches, only the LAST batch's per-env tracking / pred_emb /
            # z_ref survive → debug video shows blanks for envs 0-24, data
            # for envs 25-49. The single-view eval assumes batch_size=1 so never hits this.
            #
            # Fix: ``MPCCostModelDebug._env_breakdowns`` accumulates per-env
            # copies keyed by ref_idx across all batches. We reconstruct
            # per-env arrays here and overwrite debug_logger's latest slot.
            #
            # Activation: requires both ``debug_video=true`` AND
            # ``solver=mpc_multiview_debug`` (swaps in MPCCostModelDebug). With
            # base solver, ``_env_breakdowns`` absent → getattr empty dict →
            # ``if per_env_breakdowns:`` False → entire block skipped.
            # =================================================================
            per_env_breakdowns = getattr(solver.cost_model, "_env_breakdowns", {})
            if per_env_breakdowns:
                per_track = np.zeros(n_envs, dtype=np.float64)
                per_term = np.zeros(n_envs, dtype=np.float64)
                per_act = np.zeros(n_envs, dtype=np.float64)
                per_env_pred = None
                per_env_z_ref = None
                for env_i in range(n_envs):
                    bd = per_env_breakdowns.get(env_i)
                    if bd is None:
                        continue
                    per_track[env_i] = float(bd["tracking"].mean(dim=1).cpu().numpy()[0])
                    per_term[env_i] = float(bd["terminal"].mean(dim=1).cpu().numpy()[0])
                    per_act[env_i] = float(bd["action"].mean(dim=1).cpu().numpy()[0])
                    if "pred_emb" in bd:
                        pe = bd["pred_emb"][:, 0].cpu()  # (1, T, D)
                        if per_env_pred is None:
                            per_env_pred = torch.zeros(n_envs, *pe.shape[1:])
                        per_env_pred[env_i] = pe.squeeze(0)
                    if "z_ref" in bd:
                        zr = bd["z_ref"].cpu()  # (T, D)
                        if per_env_z_ref is None:
                            per_env_z_ref = torch.zeros(n_envs, *zr.shape)
                        per_env_z_ref[env_i] = zr
                if debug_logger.tracking_costs:
                    debug_logger.tracking_costs[-1] = per_track
                    debug_logger.terminal_costs[-1] = per_term
                    debug_logger.action_costs[-1] = per_act
                if per_env_pred is not None and debug_logger.pred_embeddings:
                    debug_logger.pred_embeddings[-1] = per_env_pred
                if per_env_z_ref is not None and debug_logger.ref_embeddings:
                    debug_logger.ref_embeddings[-1] = per_env_z_ref

        # Record per-step metrics
        if world.rewards is not None:
            rewards[:, t] = world.rewards
            distances[:, t] = -world.rewards

        # Track first success
        newly_succeeded = world.terminateds & ~episode_successes
        first_success_step[newly_succeeded] = t
        episode_successes |= world.terminateds

        # Prevent auto-reset
        world.envs.unwrapped._autoreset_envs = np.zeros((n_envs,))

    video_frames[:, -1] = world.infos["pixels"][:, -1]
    if record_wrist:
        wrist_frames[:, -1] = _to_hwc_uint8(world.infos["wrist_pixels"][:, -1])

    # -- Save videos -----------------------------------------------------------
    video_path_obj = Path(video_path)
    video_path_obj.mkdir(parents=True, exist_ok=True)

    # Frame number overlay helper (PIL fallback for environments without cv2).
    try:
        from PIL import Image, ImageDraw, ImageFont
        _font = None
        try:
            _font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        except Exception:
            _font = ImageFont.load_default()
        def _annotate(arr, t):
            img = Image.fromarray(arr)
            d = ImageDraw.Draw(img)
            txt = f"t={t:03d}"
            # Black outline + yellow fill for legibility
            for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                d.text((10 + dx, 6 + dy), txt, fill=(0, 0, 0), font=_font)
            d.text((10, 6), txt, fill=(255, 255, 0), font=_font)
            return np.array(img)
    except Exception:
        def _annotate(arr, t):
            return arr  # no PIL → silently skip overlay

    for i in range(n_envs):
        target_len_i = len(target_frames[i])
        out = imageio.get_writer(
            video_path_obj / f"rollout_{i}.mp4", fps=15, codec="libx264"
        )
        for t in range(eval_budget):
            agent_frame = video_frames[i, t].copy()
            expert_frame = target_frames[i][t % target_len_i].copy()
            top_row = np.hstack([agent_frame, expert_frame])
            if record_wrist:
                # Add wrist view next to a blank pad so total width matches top_row
                wrist_frame = wrist_frames[i, t].copy()
                pad = np.zeros_like(expert_frame)
                bottom_row = np.hstack([wrist_frame, pad])
                # Both rows same width as top_row by construction (agent+expert
                # = wrist+pad if wrist size == agent size). If sizes differ,
                # crop/pad bottom_row width to top_row width.
                if bottom_row.shape[1] != top_row.shape[1]:
                    if bottom_row.shape[1] < top_row.shape[1]:
                        diff = top_row.shape[1] - bottom_row.shape[1]
                        bottom_row = np.pad(bottom_row, ((0, 0), (0, diff), (0, 0)))
                    else:
                        bottom_row = bottom_row[:, : top_row.shape[1]]
                stacked = np.vstack([top_row, bottom_row])
            else:
                stacked = top_row
            stacked = _annotate(stacked, t)
            out.append_data(stacked)
        out.close()
    print(f"Videos saved to {video_path_obj}  (with frame# overlay{', wrist view' if record_wrist else ''})")

    # -- Debug videos ----------------------------------------------------------
    if debug_logger is not None:
        from latent_modules.mpc.debug_video import render_debug_video
        render_debug_video(
            debug_logger=debug_logger,
            video_frames=video_frames,
            target_frames=target_frames,
            references=solver._references,
            env_ref_indices=solver._env_ref_indices,
            rewards=rewards,
            episode_successes=episode_successes,
            video_path=str(video_path_obj),
            model=model,
            fps=10,
        )

    # -- Compile results -------------------------------------------------------
    success_rate = float(np.sum(episode_successes)) / n_envs * 100.0

    for i in range(n_envs):
        ep = episodes_idx[i]
        st = start_steps[i]
        agent_steps = int(first_success_step[i]) if episode_successes[i] else eval_budget
        status = "SUCCESS" if episode_successes[i] else "FAIL"
        print(f"  env {i}: ep={ep}, start={st}, {status}, agent={agent_steps} steps")

    print(f"\nSuccess rate: {success_rate:.1f}%")

    # -- Behavioral success metrics (3-criterion decomposition) ----------------
    behavioral_results = []
    for env_i in range(n_envs):
        m = _compute_episode_metrics(
            states=episode_states_per_env[env_i],
            goal_pos=goal_pos_per_env[env_i],
            table_z=table_z_per_env[env_i],
        )
        m['freeze_step'] = int(freeze_step_per_env[env_i])
        behavioral_results.append(m)

    n_grasped = sum(m['grasped'] for m in behavioral_results)
    n_no_drop = sum(m['no_drop'] for m in behavioral_results)
    n_reached = sum(m['reached_goal'] for m in behavioral_results)
    n_full = sum(m['success_full'] for m in behavioral_results)
    mean_dist = float(np.mean([m['final_dist'] for m in behavioral_results]))

    print("\n=== Behavioral metrics (per-env) ===")
    for env_i, m in enumerate(behavioral_results):
        flags = (
            ("G" if m['grasped'] else "-")
            + ("T" if m['no_drop'] else "-")
            + ("R" if m['reached_goal'] else "-")
        )
        fs = m.get('freeze_step', -1)
        fs_str = f"  ❄freeze@{fs}" if fs >= 0 else ""
        print(f"  env {env_i}: ep={episodes_idx[env_i]}  [{flags}]  "
              f"final_dist={m['final_dist']:.3f}m  "
              f"grasp_run={m['longest_grasp_run']}step{fs_str}")

    print(f"\n=== Behavioral aggregate (n={n_envs}) ===")
    print(f"  Grasped (stable lift + closed)   : {n_grasped}/{n_envs} = {100*n_grasped/n_envs:.1f}%")
    print(f"  No-drop after grasp              : {n_no_drop}/{n_envs} = {100*n_no_drop/n_envs:.1f}%")
    print(f"  Reached goal (<5cm at end)       : {n_reached}/{n_envs} = {100*n_reached/n_envs:.1f}%")
    print(f"  Full (G + T + R)                 : {n_full}/{n_envs} = {100*n_full/n_envs:.1f}%")
    print(f"  Mean final cube→goal distance    : {mean_dist:.3f}m")

    # -- RolloutLogger finalize ------------------------------------------------
    if logger is not None:
        try:
            logger.finalize(
                video_frames=video_frames,
                rewards=rewards,
                distances=distances,
                behavioral_aggregate={
                    "grasped": n_grasped, "no_drop": n_no_drop,
                    "reached_goal": n_reached, "full": n_full,
                    "mean_final_dist": mean_dist, "n_envs": n_envs,
                },
                behavioral_per_env=behavioral_results,
                primary_success_key="gtr" if "gtr" in {
                    s.name for s in logger.success_specs
                } else "naive",
            )
        except Exception as exc:
            print(f"[RolloutLogger] WARN: finalize failed: {exc!r}")

    return {
        "success_rate": success_rate,
        "episode_successes": episode_successes,
        "first_success_step": first_success_step,
        "distances": distances,
        "rewards": rewards,
        "behavioral": behavioral_results,
        "behavioral_aggregate": {
            "grasped": n_grasped, "no_drop": n_no_drop,
            "reached_goal": n_reached, "full": n_full,
            "mean_final_dist": mean_dist, "n_envs": n_envs,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Observation / action scalers
# ---------------------------------------------------------------------------

def _scaler_cache_path(cfg, stats_name: str) -> Path:
    """Where scripts/data/cache_scaler_stats.py stores the scalers of a stats dataset."""
    root = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    return root / "stats" / f"{stats_name}.npz"


def _fit_scaler(col_data: np.ndarray) -> preprocessing.StandardScaler:
    """StandardScaler fitted on one dataset column, rows containing NaN dropped."""
    processor = preprocessing.StandardScaler()
    col_data = col_data[~np.isnan(col_data).any(axis=1)]
    processor.fit(col_data)
    return processor


def _load_cached_scalers(path: Path, cols) -> dict | None:
    """Scalers saved by scripts/data/cache_scaler_stats.py, or None when the file is
    missing or lacks one of ``cols``. EPCC_NO_STATS_CACHE=1 forces a refit."""
    if os.environ.get("EPCC_NO_STATS_CACHE") == "1" or not path.is_file():
        return None
    out = {}
    with np.load(path) as z:
        if not all(f"{c}/mean_" in z.files for c in cols):
            return None
        for c in cols:
            sc = preprocessing.StandardScaler()
            sc.mean_ = z[f"{c}/mean_"]
            sc.var_ = z[f"{c}/var_"]
            sc.scale_ = z[f"{c}/scale_"]
            sc.n_samples_seen_ = z[f"{c}/n_samples_seen_"][()]
            sc.n_features_in_ = int(z[f"{c}/n_features_in_"])
            out[c] = sc
    return out


def _build_scalers(cfg, dataset) -> dict:
    """Fit (or load) the scaler of every non-image column in ``cfg.dataset.keys_to_cache``.

    Config `dataset.stats` specifies the stats-source dataset; default falls back to the
    eval dataset (only correct when eval_dataset == training_dataset). Mirrors
    eval_mpc.py:1420-1426. When the stats dataset is a separate (large) training set,
    the scalers precomputed by scripts/data/cache_scaler_stats.py are used if present.
    """
    stats_name = cfg.dataset.get("stats", cfg.eval.dataset_name)
    cols = [c for c in cfg.dataset.keys_to_cache if c not in ("pixels", "wrist_pixels")]
    cached = None
    if stats_name and stats_name != cfg.eval.dataset_name:
        path = _scaler_cache_path(cfg, stats_name)
        cached = _load_cached_scalers(path, cols)
        if cached is not None:
            print(f"[scaler] cached statistics of {stats_name}: {path}")
        else:
            print(f"[scaler] fitting on stats dataset: {stats_name}  "
                  f"(eval dataset: {cfg.eval.dataset_name})")
            stats_dataset = get_dataset(cfg, stats_name)
    else:
        stats_dataset = dataset

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels", "wrist_pixels"]:
            continue
        if cached is not None:
            processor = cached[col]
        else:
            processor = _fit_scaler(stats_dataset.get_col_data(col))
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]
        print(f"[scaler] {col}: mean={processor.mean_.round(4).tolist()}  "
              f"std={processor.scale_.round(4).tolist()}")
    return process


def _setup_eval_world(cfg) -> SimpleNamespace:
    """Build dataset / episode selection / world / transforms / scaler ``process``.

    Extracted from ``run()`` so ``eval_mpcc_multiview.py`` can reuse the same
    setup (see ``project_mpcc_mv_migration_plan`` memory). MPC behavior is
    byte-equivalent — ``run()`` calls this and unpacks the namespace.
    """
    # -- Load dataset ----------------------------------------------------------
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    print(f"Dataset: {dataset.h5_path}")
    print(f"Policy: {cfg.policy}")

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    episode_lengths = get_episodes_length(dataset, ep_indices)
    ep_len_map = dict(zip(ep_indices, episode_lengths))

    goal_offset = cfg.eval.goal_offset_steps
    eval_budget = cfg.eval.eval_budget

    # -- Optional filter: keep only physically-successful expert episodes.
    # Eval-on-success-only is the apples-to-apples mode: the expert demonstrated
    # the task, so any MPC failure is a planner regression rather than a dataset
    # artefact. Gated by ``+eval.filter_successful_only=true``; default off
    # preserves the existing all-episode sampling pool.
    filter_successful = bool(cfg.eval.get("filter_successful_only", False))
    if filter_successful:
        if "ep_physical_success" not in dataset.column_names:
            raise ValueError(
                f"eval.filter_successful_only=true but dataset "
                f"{cfg.eval.dataset_name} has no 'ep_physical_success' column."
            )
        # Per-episode bool array (length = num_episodes); ep_id indexes directly.
        success_arr = np.asarray(
            dataset.get_col_data("ep_physical_success")
        ).astype(bool)
        success_mask = np.array(
            [bool(success_arr[int(e)]) for e in ep_indices], dtype=bool
        )
        n_total = len(ep_indices)
        ep_indices = ep_indices[success_mask]
        print(f"[episodes] filter_successful_only=true: "
              f"{len(ep_indices)}/{n_total} expert-successful episodes retained")

    # -- Episode selection -----------------------------------------------------
    # ``eval.episodes`` override (e.g., +eval.episodes=[0,1,...,9]) forces a
    # specific list — useful for paired-trajectory comparisons across datasets
    # (e.g., D1 c=0.3 ↔ D3 c=1.0 same trajectory at different speeds).
    # When unset, fall back to seeded random sampling of n_eval episodes.
    g = np.random.default_rng(cfg.seed)
    override_episodes = cfg.eval.get("episodes", None)
    # -- Optional: scene subtask SEGMENT selection ----------------------------
    # ``+eval.segment=BC`` (scene-only) restricts the rollout to a contiguous run
    # of the demo's 5 subtasks (A open_drawer, B cube_into_drawer, C close_drawer,
    # D press_window_btn, E open_window). The env is initialised at the demo frame
    # where the *preceding* stage completed and the reference is cut at the frame
    # where the segment's last stage completes — so a segment run is the same
    # protocol as a full run, just at a different [start, end] index pair.
    #
    # Frames come from the offline sidecar produced by
    # ``scripts/baseline/label_scene_stages.py``, which recomputes the collector's
    # own physical predicates from stored privileged columns.
    #
    # Generalises ``+eval.start_at_drawer_open`` (which is exactly segment=BCDE
    # with an uncut tail) and is mutually exclusive with it and with start_offset.
    _STAGES = "ABCDE"
    segment = cfg.eval.get("segment", None)
    seg_start_by_ep = seg_end_by_ep = None
    if segment is not None:
        segment = str(segment).upper().strip()
        if not segment or any(c not in _STAGES for c in segment):
            raise ValueError(f"+eval.segment={segment!r} must be drawn from {_STAGES}")
        s_idx = [_STAGES.index(c) for c in segment]
        if s_idx != list(range(s_idx[0], s_idx[0] + len(s_idx))):
            raise ValueError(
                f"+eval.segment={segment!r} must be a contiguous ascending run "
                f"(e.g. A, BC, BCD) — got out-of-order or gapped stages."
            )
        labels_path = cfg.eval.get("stage_labels", None)
        if labels_path is None:
            raise ValueError(
                "+eval.segment requires +eval.stage_labels=<path to the .npz "
                "written by scripts/baseline/label_scene_stages.py>."
            )
        _lab = np.load(str(labels_path), allow_pickle=False)
        # Guard against pairing labels with a different dataset — the frame
        # indices are only meaningful for the h5 they were computed from.
        _meta = json.loads(str(_lab["meta"]))
        if Path(_meta["h5"]).name != Path(str(dataset.h5_path)).name:
            raise ValueError(
                f"+eval.stage_labels was built from {_meta['h5']} but the eval "
                f"dataset is {dataset.h5_path}; stage frames would be wrong."
            )
        _stage_end = _lab["stage_end"]      # (n_ep, 5), -1 where unreached
        _stage_start = _lab["stage_start"]  # (n_ep, 5) = stage_end shifted, 0 for A
        first_i, last_i = s_idx[0], s_idx[-1]
        seg_start_by_ep = _stage_start[:, first_i]
        seg_end_by_ep = _stage_end[:, last_i]
        # An episode is usable iff the demo actually reached the segment's end
        # (which, by chaining, implies every stage in the segment was performed)
        # and leaves a non-empty window.
        seg_ok = (seg_end_by_ep >= 0) & (seg_end_by_ep > seg_start_by_ep)
        if first_i > 0:
            seg_ok &= _stage_start[:, first_i] > 0
        n_before = len(ep_indices)
        _keep = np.array([bool(seg_ok[int(e)]) for e in ep_indices], dtype=bool)
        ep_indices = ep_indices[_keep]
        if len(ep_indices) == 0:
            raise ValueError(
                f"+eval.segment={segment}: no episode in the pool completes this "
                f"segment (checked {n_before})."
            )
        _d = seg_end_by_ep[ep_indices] - seg_start_by_ep[ep_indices]
        print(f"[segment] {segment}: {len(ep_indices)}/{n_before} episodes usable; "
              f"window length median={int(np.median(_d))} "
              f"p10={int(np.percentile(_d, 10))} p90={int(np.percentile(_d, 90))} "
              f"max={int(_d.max())}")

    if override_episodes is not None:
        eval_episodes = np.array(list(override_episodes), dtype=np.int64)
        missing = [int(e) for e in eval_episodes if int(e) not in ep_len_map]
        if missing:
            raise ValueError(
                f"+eval.episodes contains episodes {missing} not present in "
                f"dataset {cfg.eval.dataset_name} (has {len(ep_indices)} episodes)"
            )
        if filter_successful:
            non_success = [int(e) for e in eval_episodes if int(e) not in ep_indices]
            if non_success:
                raise ValueError(
                    f"+eval.episodes contains non-successful episodes "
                    f"{non_success} which conflict with "
                    f"+eval.filter_successful_only=true"
                )
        n_eval = len(eval_episodes)
        print(f"[episodes] override: {eval_episodes.tolist()} (n_eval={n_eval})")
    else:
        n_eval = min(cfg.eval.num_eval, len(ep_indices))
        eval_episodes = g.choice(ep_indices, size=n_eval, replace=False).astype(np.int64)

    full = cfg.get("full", False)
    # `eval.start_offset` (optional, default None): fixed env-init frame for all
    # envs — overrides the random-start-frame default. Useful for ablation
    # (e.g. start_offset=5 puts env init at dataset frame 5 = train-distribution
    # anchor for diffusion-MPC, dodging the t=0 OOD cond). Same for all envs.
    start_offset = cfg.eval.get("start_offset", None)

    # ``+eval.start_at_drawer_open=true`` (scene-only): pick a per-episode start
    # frame at the moment the drawer is fully opened in the demo, skipping the
    # open-drawer subtask. Lets MPC dogfood the cube-transport phase without
    # the upstream drawer bottleneck. Threshold mirrors the env's internal
    # `qpos < -0.12` check (scene_env.py:820); the dataset column
    # ``privileged_drawer_pos`` is the drawer_slide joint qpos
    # (manipspace scene_env.py:586). ``start_offset`` and ``start_at_drawer_open``
    # are mutually exclusive.
    start_at_drawer_open = bool(cfg.eval.get("start_at_drawer_open", False))
    drawer_open_threshold = float(cfg.eval.get("drawer_open_threshold", -0.12))
    per_ep_start_idx = None
    per_ep_end_idx = None
    if segment is not None:
        if start_offset is not None or start_at_drawer_open:
            raise ValueError(
                "+eval.segment is mutually exclusive with +eval.start_offset and "
                "+eval.start_at_drawer_open (all three pin the env-init frame)."
            )
        per_ep_start_idx = np.array(
            [int(seg_start_by_ep[int(e)]) for e in eval_episodes], dtype=np.int64
        )
        per_ep_end_idx = np.array(
            [int(seg_end_by_ep[int(e)]) for e in eval_episodes], dtype=np.int64
        )
        # Segment A starts at frame 0, which is the OOD diffusion conditioning
        # that `eval.start_offset=5` exists to dodge on full-task runs. Without
        # this pad the A cell would be confounded against every other segment
        # (which starts mid-episode and is therefore in-distribution).
        seg_pad = int(cfg.eval.get("segment_start_offset", 0))
        if seg_pad:
            room = per_ep_end_idx - per_ep_start_idx
            applied = np.minimum(seg_pad, np.maximum(room - 1, 0))
            per_ep_start_idx = per_ep_start_idx + applied
            print(f"[segment] segment_start_offset={seg_pad} applied "
                  f"(min {int(applied.min())}, max {int(applied.max())})")
        print(f"[segment] {segment}: per-episode [start, end] = "
              + ", ".join(
                  f"ep{int(e)}[{int(s)},{int(t)}]"
                  for e, s, t in zip(
                      eval_episodes[:8], per_ep_start_idx[:8], per_ep_end_idx[:8]
                  )
              )
              + (" ..." if n_eval > 8 else ""))
    if start_at_drawer_open:
        if start_offset is not None:
            raise ValueError(
                "+eval.start_at_drawer_open and +eval.start_offset are mutually "
                "exclusive (both pin the env-init frame)."
            )
        if "privileged_drawer_pos" not in dataset.column_names:
            raise ValueError(
                "+eval.start_at_drawer_open=true requires "
                "'privileged_drawer_pos' column (scene-task dataset)."
            )
        drawer_pos_all = np.asarray(
            dataset.get_col_data("privileged_drawer_pos")
        )
        per_ep_start_idx = np.zeros(n_eval, dtype=np.int64)
        no_open_eps: list[int] = []
        for i, ep in enumerate(eval_episodes):
            ep_id = int(ep)
            off = int(dataset.offsets[ep_id])
            ln = int(dataset.lengths[ep_id])
            qpos = drawer_pos_all[off : off + ln]
            qpos = qpos[..., 0] if qpos.ndim > 1 else qpos
            matches = np.where(qpos < drawer_open_threshold)[0]
            if len(matches) == 0:
                no_open_eps.append(ep_id)
                per_ep_start_idx[i] = 0
            else:
                per_ep_start_idx[i] = int(matches[0])
        if no_open_eps:
            raise ValueError(
                f"start_at_drawer_open: episodes {no_open_eps} never reach "
                f"drawer_pos < {drawer_open_threshold} — dataset doesn't have "
                f"a clean open-drawer phase."
            )
        print(f"[eval] start_at_drawer_open=true "
              f"(threshold qpos<{drawer_open_threshold}): per-episode "
              f"start frames = {per_ep_start_idx.tolist()}")

    if full:
        goal_offsets = np.array([int(ep_len_map[ep]) - 1 for ep in eval_episodes])
        if per_ep_end_idx is not None:
            # Segment mode: BOTH ends are pinned. The reference/goal is cut at
            # the frame where the segment's last stage completes — not at
            # ep_len-1 — so the tracker is never asked to chase the rest of the
            # demo (which would reintroduce the downstream stages we excluded).
            goal_offsets = per_ep_end_idx - per_ep_start_idx
            assert (goal_offsets > 0).all(), (
                f"+eval.segment={segment}: non-positive window for episodes "
                f"{[int(e) for e, d in zip(eval_episodes, goal_offsets) if d <= 0]}."
            )
            eval_start_idx = per_ep_start_idx.copy()
        elif per_ep_start_idx is not None:
            # Per-episode start; relative goal_offset shrinks per episode so
            # absolute end stays at ep_len-1.
            goal_offsets = goal_offsets - per_ep_start_idx
            assert (goal_offsets > 0).all(), (
                f"start_at_drawer_open: some per-episode start frames "
                f"≥ ep_len-1 — drawer-open detected at the very last frame "
                f"of those episodes."
            )
            eval_start_idx = per_ep_start_idx.copy()
        elif start_offset is not None:
            # Shift the [start, ep_len-1] window by start_offset; relative
            # goal_offset shrinks so absolute end_step stays at ep_len-1.
            so = int(start_offset)
            goal_offsets = goal_offsets - so
            assert (goal_offsets > 0).all(), (
                f"start_offset={so} too large for some episodes (min ep_len="
                f"{int(min(ep_len_map[e] for e in eval_episodes))}); window inverted."
            )
            eval_start_idx = np.full(n_eval, so, dtype=np.int64)
        else:
            eval_start_idx = np.zeros(n_eval, dtype=np.int64)
        goal_offset = int(goal_offsets.max())  # for eval_budget / world config
        # New eval_budget semantics (post-task_name refactor):
        #   - None  → auto-derive from dataset (use goal_offset directly).
        #   - int   → user override, used as-is (no max-grow against goal_offset).
        # Legacy max-grow behavior is recovered by leaving eval_budget as None
        # in the task default (config/eval/{ws,sh}_reproduce/task/*.yaml).
        if eval_budget is None:
            eval_budget = goal_offset
            # A full-task run gets free slack because ep_len (700) exceeds the
            # frame the demo actually finishes on (~560). Cutting a segment
            # removes that slack: the budget becomes the demo's own duration, so
            # a controller even slightly slower than the oracle fails on time
            # rather than on capability. Restore comparable headroom explicitly.
            if segment is not None:
                _scale = float(cfg.eval.get("segment_budget_scale", 1.0))
                if _scale != 1.0:
                    eval_budget = int(np.ceil(_scale * goal_offset))
                    print(f"[segment] eval_budget {goal_offset} × "
                          f"segment_budget_scale={_scale} → {eval_budget}")
            print(f"[full-mode] eval_budget auto-derived from ep_len → {eval_budget}")
        else:
            print(f"[full-mode] eval_budget user-override → {eval_budget} "
                  f"(derived goal_offset={goal_offset})")
    else:
        if per_ep_end_idx is not None:
            raise ValueError(
                "+eval.segment requires full=true — the segment pins both the "
                "start and the end frame, whereas full=false imposes a fixed "
                "goal_offset_steps window that would silently truncate or "
                "overrun the segment."
            )
        goal_offsets = np.full(n_eval, goal_offset)
        if per_ep_start_idx is not None:
            max_safe = np.array(
                [int(ep_len_map[ep]) - goal_offset - 1 for ep in eval_episodes]
            )
            over = [
                (int(eval_episodes[i]), int(per_ep_start_idx[i]), int(max_safe[i]))
                for i in range(n_eval) if per_ep_start_idx[i] > max_safe[i]
            ]
            assert not over, (
                f"start_at_drawer_open: drawer-open frame past max_safe for "
                f"episodes (ep, start, max_safe)={over}; need full=true or "
                f"smaller goal_offset_steps."
            )
            eval_start_idx = per_ep_start_idx.copy()
        elif start_offset is not None:
            so = int(start_offset)
            min_ep_len = int(min(ep_len_map[e] for e in eval_episodes))
            max_safe = min_ep_len - goal_offset - 1
            assert so <= max_safe, (
                f"start_offset={so} > max_safe={max_safe} (min ep_len={min_ep_len}, "
                f"goal_offset={goal_offset}); would push end past ep boundary."
            )
            eval_start_idx = np.full(n_eval, so, dtype=np.int64)
        else:
            eval_start_idx = np.array([
                g.integers(0, max(int(ep_len_map[ep]) - goal_offset - 1, 1))
                for ep in eval_episodes
            ])
    if start_offset is not None:
        print(f"[eval] start_offset={int(start_offset)} (fixed, ablation mode — "
              f"all envs init at dataset frame {int(start_offset)})")

    # -- Create world ----------------------------------------------------------
    cfg.world.max_episode_steps = 2 * eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # -- Transforms and preprocessing ------------------------------------------
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
        "wrist_pixels": img_transform(cfg),
        "goal_wrist_pixels": img_transform(cfg),
    }

    process = _build_scalers(cfg, dataset)

    return SimpleNamespace(
        dataset=dataset,
        world=world,
        transform=transform,
        process=process,
        eval_episodes=eval_episodes,
        eval_start_idx=eval_start_idx,
        goal_offsets=goal_offsets,
        goal_offset=goal_offset,
        eval_budget=eval_budget,
        n_eval=n_eval,
        segment=segment,
    )


def _setup_eval_solver(
    cfg,
    eval_world,
    *,
    gt_refs: bool = True,
    policy_cls=None,
    policy_kwargs: dict | None = None,
):
    """Build (model, solver, policy, plan_config) for MV MPC eval.

    Shared between ``eval_mpc_multiview.run()`` (gt_refs=True, default
    MultiviewWorldModelPolicy) and ``eval_diffusion_mpc.run()`` (gt_refs=False,
    DiffusionMPCPolicy). Mirrors ``_setup_eval_solver_mpcc`` in
    eval_mpcc_multiview.py.

    For random policy, returns ``(None, None, RandomPolicy(), None)``.

    Args:
        cfg: Hydra config (must have cfg.policy, cfg.solver, cfg.plan_config)
        eval_world: SimpleNamespace from ``_setup_eval_world``.
        gt_refs: If True (default), load dataset GT references via
            ``solver.set_reference_episodes`` + handle ext_init/GT_INIT.
            If False, caller must populate ``solver._references`` and
            ``solver._env_ref_indices`` BEFORE ``world.set_policy(policy)``.
        policy_cls: Policy class. Default ``MultiviewWorldModelPolicy``.
        policy_kwargs: Extra kwargs merged with the standard
            ``{solver, config, process, transform}`` policy constructor args.

    Returns:
        (model, solver, policy, plan_config). First three are None for
        random policy.
    """
    if cfg.get("policy", "random") == "random":
        return None, None, swm.policy.RandomPolicy(), None

    model = swm.policy.AutoCostModel(cfg.policy)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    # HF ViT compat patch (transformers 4.46.x): ViTSelfAttention.dropout missing
    # + ViTConfig default attributes privatised. Standard fix mirrored from
    # cache_latents.py / train_decoder.py / validate_diffusion_planner.py.
    # Centralised here so all callers (eval_mpc_multiview, eval_diffusion_mpc)
    # inherit the patch automatically.
    import transformers as _hf_transformers
    for _m in model.modules():
        if hasattr(_m, "config") and isinstance(_m.config, _hf_transformers.PretrainedConfig):
            _fresh = type(_m.config)()
            for _k, _v in _fresh.__dict__.items():
                if _k not in _m.config.__dict__:
                    setattr(_m.config, _k, _v)
        if "ViTSelfAttention" in type(_m).__name__ and not hasattr(_m, "dropout"):
            _m.dropout = torch.nn.Dropout(getattr(_m, "dropout_prob", 0.0))

    plan_config = swm.PlanConfig(**OmegaConf.to_container(cfg.plan_config, resolve=True))
    solver = hydra.utils.instantiate(
        cfg.solver, model=model, transform=eval_world.transform["pixels"]
    )

    # Expose dataset scalers to the solver so _encode_episode can apply them
    # to proprio columns during reference-trajectory encoding. Runtime
    # per-step policy.get_action already applies the same scalers via
    # WorldModelPolicy._prepare_info — keeps ref-path parity with runtime
    # for AP (no-op for AW since no proprio keys).
    solver._scalers = {k: v for k, v in eval_world.process.items() if k != "action"}

    if gt_refs:
        # Build per-episode reference trajectories
        solver.set_reference_episodes(
            eval_world.eval_episodes, eval_world.eval_start_idx, eval_world.goal_offsets
        )

        # -- External GT-init from npz (e.g., produced by another MPC run with
        # --    +eval.save_data=true). Path = ``+eval.init_action_npz=<path>``.
        # -- npz["actions"] shape: (S, n_envs, H, A_phys). Solver pops one
        # -- slice per solve(); falls through to cold-start when exhausted.
        ext_init_path = cfg.eval.get("init_action_npz", None)
        if ext_init_path:
            data = np.load(ext_init_path, allow_pickle=False)
            ext_actions = torch.from_numpy(data["actions"]).float()
            assert ext_actions.dim() == 4, f"actions must be (S, n_envs, H, A); got {ext_actions.shape}"
            assert ext_actions.shape[1] == len(eval_world.eval_episodes), (
                f"npz n_envs={ext_actions.shape[1]} ≠ eval_episodes={len(eval_world.eval_episodes)}"
            )
            solver._gt_init_actions = ext_actions
            print(f"[ext GT-init] loaded {ext_init_path}  "
                  f"shape={tuple(ext_actions.shape)} (S={ext_actions.shape[0]} solves)")

        # -- GT_INIT: inject dataset GT actions as CEM mean (WM vs CEM decomposition) --
        init_from_gt = cfg.eval.get("init_from_gt", False)
        if init_from_gt:
            print(f"[GT-init test] Loading per-env GT actions from dataset...")
            gt_actions_per_env = []
            for i, ep_idx in enumerate(eval_world.eval_episodes):
                start = int(eval_world.eval_start_idx[i])
                length = (int(eval_world.goal_offsets[i])
                          if hasattr(eval_world.goal_offsets, "__len__")
                          else int(eval_world.goal_offsets))
                ep_data = eval_world.dataset.load_chunk(
                    np.array([int(ep_idx)]),
                    np.array([start]),
                    np.array([start + length + 1]),
                )[0]
                raw_action = ep_data["action"]
                act_tensor = (raw_action.float() if isinstance(raw_action, torch.Tensor)
                              else torch.from_numpy(np.array(raw_action)).float())
                n_nan = torch.isnan(act_tensor).any(dim=-1).sum().item()
                act_tensor = torch.nan_to_num(act_tensor, nan=0.0)
                gt_actions_per_env.append(act_tensor)
                print(f"  env {i}: ep={ep_idx}, start={start}, "
                      f"actions len={act_tensor.shape[0]} (replaced {n_nan} NaN rows with 0)")

            _orig_solver_solve = solver.solve
            action_block = int(cfg.plan_config.action_block)
            horizon = int(cfg.plan_config.horizon)
            physical_action_dim = int(eval_world.world.envs.unwrapped.envs[0].action_space.shape[0])
            flat_action_dim = action_block * physical_action_dim
            needed_env_frames = horizon * action_block
            process = eval_world.process
            eval_episodes_local = eval_world.eval_episodes

            def _solve_with_gt_init(info_dict, init_action=None):
                env_ref_steps = solver._env_ref_steps
                gt_init = torch.zeros(solver._n_envs, horizon, flat_action_dim)
                for i in range(solver._n_envs):
                    start = int(env_ref_steps[i])
                    gt_ep = gt_actions_per_env[i]
                    ep_len = gt_ep.shape[0]
                    end = min(start + needed_env_frames, ep_len)
                    segment = gt_ep[start:end]
                    if segment.shape[0] < needed_env_frames:
                        pad = torch.zeros(needed_env_frames - segment.shape[0],
                                          physical_action_dim)
                        segment = torch.cat([segment, pad], dim=0)
                    gt_init[i] = segment.reshape(horizon, flat_action_dim)
                gt_init = torch.nan_to_num(gt_init, nan=0.0).clamp(-1.0, 1.0)

                # env → normalized space (match CEM's space)
                if 'action' in process:
                    _scaler = process['action']
                    _shape = gt_init.shape
                    _flat = gt_init.reshape(-1, physical_action_dim).numpy()
                    _flat_norm = _scaler.transform(_flat)
                    gt_init = torch.from_numpy(_flat_norm).float().reshape(_shape)

                outputs = _orig_solver_solve(info_dict, init_action=gt_init)

                # Per-env GT-drift diagnostic (mirrors eval_mpc.py SV version).
                # Shows whether CEM stayed near GT or drifted to a predictor-
                # hallucination minimum. All values in NORMALIZED space.
                raw_actions = outputs["actions"].cpu()
                receding = int(cfg.plan_config.receding_horizon)
                for i in range(solver._n_envs):
                    gt_i = gt_init[i]
                    cem_i = raw_actions[i]
                    delta_i = (cem_i - gt_i).abs()
                    gt_mag = gt_i.abs().mean().item()
                    cem_mag = cem_i.abs().mean().item()
                    drift = delta_i.mean().item()
                    max_drift = delta_i.max().item()
                    n_sat = (cem_i.abs() > 1.0).float().mean().item()
                    ratio = drift / (gt_mag + 1e-8)
                    ep_idx = int(eval_episodes_local[i])
                    rs = int(solver._env_ref_steps[i])
                    print(
                        f"  [GT-init env {i} ep={ep_idx} rs={rs}] "
                        f"|GT|={gt_mag:.3f} |CEM|={cem_mag:.3f} "
                        f"|Δ|={drift:.3f} max|Δ|={max_drift:.3f} "
                        f"ratio={ratio:.3f} sat={n_sat:.0%}"
                    )

                    cem_steps = cem_i.reshape(horizon, action_block, physical_action_dim)
                    cem_steps = cem_steps.reshape(-1, physical_action_dim)
                    gt_steps = gt_i.reshape(horizon, action_block, physical_action_dim)
                    gt_steps = gt_steps.reshape(-1, physical_action_dim)
                    cem_signed = cem_steps.mean(dim=0)
                    cem_unsigned = cem_steps.abs().mean(dim=0)
                    gt_signed = gt_steps.mean(dim=0)
                    xyz_mag_cem = cem_steps[:, :3].abs().mean().item()
                    rot_grip_mag_cem = cem_steps[:, 3:].abs().mean().item()
                    xyz_mag_gt = gt_steps[:, :3].abs().mean().item()
                    rot_grip_mag_gt = gt_steps[:, 3:].abs().mean().item()
                    n_applied = min(receding * action_block, cem_steps.shape[0])
                    first_mag_cem = cem_steps[:n_applied].abs().mean().item()
                    first_signed_cem = cem_steps[:n_applied].mean(dim=0)
                    osc = (cem_unsigned - cem_signed.abs())
                    print(
                        f"      ↳ xyz |a|: GT={xyz_mag_gt:.3f}  CEM={xyz_mag_cem:.3f}  "
                        f"| rot+grip |a|: GT={rot_grip_mag_gt:.3f}  CEM={rot_grip_mag_cem:.3f}"
                    )
                    print(
                        f"      ↳ first {n_applied} env-steps |a|={first_mag_cem:.3f}  "
                        f"signed=[{', '.join(f'{v:+.2f}' for v in first_signed_cem.tolist())}]"
                    )
                    print(
                        f"      ↳ all-step signed: CEM=[{', '.join(f'{v:+.2f}' for v in cem_signed.tolist())}]"
                        f"  GT=[{', '.join(f'{v:+.2f}' for v in gt_signed.tolist())}]"
                    )
                    print(
                        f"      ↳ oscillation (unsigned − |signed|) per dim: "
                        f"[{', '.join(f'{v:.2f}' for v in osc.tolist())}]  (large = oscillating)"
                    )
                return outputs

            solver.solve = _solve_with_gt_init
            print(f"[GT-init test] solver.solve patched. CEM starts from dataset actions.")

    # MV-aware variant: pre-fills past-action buffer so upstream's pixels-only
    # warm-up fallback never fires (would mismatch with wrist_pixels / proprio_*
    # time dims in PerViewJEPA.encode).
    policy_cls_final = policy_cls or MultiviewWorldModelPolicy
    base_kwargs = {
        "solver": solver,
        "config": plan_config,
        "process": eval_world.process,
        "transform": eval_world.transform,
    }
    if policy_kwargs:
        base_kwargs.update(policy_kwargs)
    policy = policy_cls_final(**base_kwargs)

    return model, solver, policy, plan_config


@hydra.main(version_base=None, config_path="./config/eval", config_name="multiview_cube")
def run(cfg: DictConfig):
    """Run multi-view MPC evaluation."""

    _w = _setup_eval_world(cfg)
    dataset = _w.dataset
    world = _w.world
    transform = _w.transform
    process = _w.process
    eval_episodes = _w.eval_episodes
    eval_start_idx = _w.eval_start_idx
    goal_offsets = _w.goal_offsets
    goal_offset = _w.goal_offset
    eval_budget = _w.eval_budget
    n_eval = _w.n_eval

    # -- Build policy ----------------------------------------------------------
    policy_name = cfg.get("policy", "random")

    # Setup model + solver + policy via helper (shared with eval_diffusion_mpc).
    # gt_refs=True (default) → load dataset GT references via
    # set_reference_episodes + handle ext_init/GT_INIT inside the helper.
    model, solver, policy, _plan_config = _setup_eval_solver(cfg, _w)

    # Attach decoder for debug visualization if configured (after helper,
    # before world.set_policy → solver.configure).
    if model is not None:
        debug_cfg = cfg.get("debug", {})
        decoder_ckpt = debug_cfg.get("decoder_checkpoint") if debug_cfg else None
        if decoder_ckpt and cfg.eval.get("debug_video", False):
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location("root_utils", Path(__file__).parent / "utils.py")
            _root_utils = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_root_utils)
            attach_decoder_from_checkpoint = _root_utils.attach_decoder_from_checkpoint
            model = attach_decoder_from_checkpoint(model, decoder_ckpt, map_location="cuda")
            print(f"[MPC eval] Decoder loaded from {decoder_ckpt}")

    # -- Output path -----------------------------------------------------------
    policy_dir = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if policy_name != "random"
        else Path(__file__).parent
    )
    run_name = cfg.output.get("run_name") or "eval_mpc"
    results_path = policy_dir / run_name
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"[MPC eval] episodes={np.unique(eval_episodes).tolist()}, "
          f"goal_offset={goal_offset}, eval_budget={eval_budget}, n={n_eval}")

    # -- Run evaluation --------------------------------------------------------
    world.set_policy(policy)
    _enabled_views = getattr(model, "enabled_views", ("agent", "wrist")) if model is not None else ("agent",)
    _proprio_keys = tuple(getattr(model, "proprio_keys", ()) or ())
    if "wrist" in _enabled_views:
        _attach_wrist_renderers(world, img_size=cfg.world.get("width", 224))
        print(f"[renderer] wrist renderer attached (enabled_views={list(_enabled_views)})")
    else:
        print(f"[renderer] wrist renderer SKIPPED (enabled_views={list(_enabled_views)})")
    if "proprio" in _enabled_views:
        _attach_proprio_alias(world, _proprio_keys)

    # -- CEM env-space projection (scaler on inner CEM) -----------------------
    # Projection lives entirely in CEMSolver (2026-04-24 refactor). We always
    # attach the training scaler if available; CEM decides on/off via its own
    # ``env_clip_range`` + ``projection_mode`` options (see mpc_multiview.yaml).
    # To disable: ``solver.inner_solver.projection_mode=none`` or
    # ``solver.inner_solver.env_clip_range=null`` in Hydra overrides.
    inner = getattr(solver, "_inner_solver", None) if solver is not None else None
    if (inner is not None
            and hasattr(inner, "set_action_scaler")
            and hasattr(world.policy, "process")
            and "action" in world.policy.process):
        inner.set_action_scaler(world.policy.process["action"])
        rng = getattr(inner, "env_clip_range", None)
        mode = getattr(inner, "projection_mode", "state")
        if rng is not None and mode != "none":
            print(f"[cem] env-space projection ENABLED  mode={mode}  "
                  f"clip={tuple(rng)}")
        else:
            print(f"[cem] env-space projection DISABLED "
                  f"(env_clip_range={rng}, mode={mode})")
    else:
        print("[cem] scaler not attached (no inner solver / no policy.process['action'])")

    # Past-action context injection handled by upstream WorldModelPolicy
    # (stable-worldmodel-srl 47e7a3d). Activated via
    # ``plan_config.use_past_context`` which is now a native PlanConfig field.

    start_time = time.time()
    metrics = evaluate_tracking_from_dataset(
        world,
        dataset,
        solver=solver,
        episodes_idx=eval_episodes.tolist(),
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=goal_offsets if goal_offsets is not None else goal_offset,
        eval_budget=eval_budget,
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
        save_video=cfg.eval.get("save_video", True),
        debug=cfg.eval.get("debug_video", False),
        model=model if cfg.eval.get("debug_video", False) else None,
    )
    elapsed = time.time() - start_time

    # -- Summary ---------------------------------------------------------------
    sr = metrics["success_rate"]
    print(f"\n[MPC eval] Success rate: {sr:.1f}%")
    print(f"[MPC eval] Eval time: {elapsed:.1f}s")

    # Save results
    results_file = results_path / (cfg.output.get("filename") or "mpc_results.txt")
    with open(results_file, "w") as f:
        f.write(f"success_rate: {sr:.1f}%\n")
        f.write(f"eval_budget: {eval_budget}\n")
        f.write(f"goal_offset: {goal_offset}\n")
        f.write(f"n_eval: {n_eval}\n")
        f.write(f"eval_time: {elapsed:.1f}s\n")

        bag = metrics.get("behavioral_aggregate")
        if bag is not None:
            n_e = bag["n_envs"]
            f.write("\n=== Behavioral aggregate ===\n")
            f.write(f"grasped:      {bag['grasped']}/{n_e} = {100*bag['grasped']/n_e:.1f}%\n")
            f.write(f"no_drop:      {bag['no_drop']}/{n_e} = {100*bag['no_drop']/n_e:.1f}%\n")
            f.write(f"reached_goal: {bag['reached_goal']}/{n_e} = {100*bag['reached_goal']/n_e:.1f}%\n")
            f.write(f"full_GTR:     {bag['full']}/{n_e} = {100*bag['full']/n_e:.1f}%\n")
            f.write(f"mean_final_dist: {bag['mean_final_dist']:.4f}m\n")

        f.write("\n=== Per-env ===\n")
        beh = metrics.get("behavioral") or [{}] * len(metrics["episode_successes"])
        for i in range(len(metrics["episode_successes"])):
            ep_idx = int(eval_episodes[i]) if i < len(eval_episodes) else -1
            m = beh[i] if i < len(beh) else {}
            flags = (
                ("G" if m.get("grasped") else "-")
                + ("T" if m.get("no_drop") else "-")
                + ("R" if m.get("reached_goal") else "-")
            )
            freeze_step = int(m.get("freeze_step", -1))
            f.write(
                f"env_{i}: ep={ep_idx}, success={bool(metrics['episode_successes'][i])}, "
                f"first_success={int(metrics['first_success_step'][i])}, "
                f"flags=[{flags}], "
                f"final_dist={m.get('final_dist', float('nan')):.4f}m, "
                f"grasp_run={m.get('longest_grasp_run', 0)}step, "
                f"first_grasp={m.get('first_grasp_step', -1)}, "
                f"drop_step={m.get('drop_step', -1)}, "
                f"freeze_step={freeze_step}\n"
            )
    print(f"[MPC eval] Results saved to {results_file}")

    # ---- Per-solve npz dump (cfg.eval.save_data) ----
    if cfg.eval.get("save_data", False) and solver is not None and getattr(solver, "_solve_log", None):
        npz_file = results_path / "solve_log.npz"
        log = solver._solve_log
        actions = torch.stack([entry["actions"] for entry in log], dim=0)  # (S, n_envs, H, A)
        # Per-solve scalars: shape (S, n_envs)
        def _stack(field):
            return np.array([entry.get(field, []) for entry in log], dtype=np.float32)
        np.savez(
            npz_file,
            actions=actions.cpu().numpy(),     # (S, n_envs, H, A_phys)
            ref_step=np.array([entry["ref_step"] for entry in log], dtype=np.float32),
            costs=_stack("costs"),
            init_l2=_stack("init_l2"),
            final_l2=_stack("final_l2"),
            init_aabs=_stack("init_aabs"),
            final_aabs=_stack("final_aabs"),
            n_solves=len(log),
            n_envs=len(eval_episodes),
            episodes=np.array(eval_episodes, dtype=np.int64),
        )
        print(f"[MPC eval] solve_log.npz saved to {npz_file}  "
              f"(S={len(log)}, n_envs={len(eval_episodes)}, A={actions.shape[-1]})")


if __name__ == "__main__":
    run()
