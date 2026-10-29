"""Collect multiview DIFFUSION cube data: random init, FIXED goal, ends on success.

Mimics collect_diffusion_cube.py + adds eye-in-hand wrist camera.
Two-phase approach:
  1. Collect with swm.World + FixedGoalCubePolicy (random init, fixed goal);
     episode terminates right after cube reaches goal (no trailing hover).
  2. Replay qpos/qvel + mocap to render wrist camera, append wrist_pixels.

Usage:
    python collect_multiview_diffusion_cube.py mode=validate
    python collect_multiview_diffusion_cube.py mode=collect
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401
import hydra
import imageio
import mujoco
import numpy as np
from loguru import logger as logging
from omegaconf import DictConfig
from tqdm import tqdm

import stable_worldmodel as swm  # noqa: F401
from ogbench.manipspace import lie
from ogbench.manipspace.oracles.markov.cube_markov import (
    CubeMarkovOracle as _BaseCubeMarkovOracle,
)


# ---------------------------------------------------------------------------
# Wrist camera renderer (same 6DOF override as WM multiview scripts)
# ---------------------------------------------------------------------------

class WristCameraRenderer:
    """True first-person eye-in-hand via direct scene.camera override."""

    def __init__(self, model, height=224, width=224):
        self.model = model
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultFreeCamera(model, self.cam)

        self._site_id = None
        for i in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name and 'attachment_site' in name:
                self._site_id = i
                break
        if self._site_id is None:
            logging.warning('attachment_site not found')

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

    def close(self):
        if hasattr(self, 'renderer'):
            self.renderer.close()


# ---------------------------------------------------------------------------
# Oracle + policy (from collect_diffusion_cube.py)
# ---------------------------------------------------------------------------

class CubeMarkovOracle(_BaseCubeMarkovOracle):
    def reset(self, ob, info):
        max_step = self._max_step
        super().reset(ob, info)
        self._max_step = max_step


class FixedGoalCubePolicy:
    """Expert policy: random init (in bounds) + fixed goal; ends when oracle done."""

    def __init__(self, env, init_bounds, goal_xyz, slowdown_factor=1.0,
                 action_noise=0.0, min_norm=0.4):
        self.env = env
        self.c = float(slowdown_factor)
        self.action_noise = float(action_noise)
        self.min_norm = float(min_norm)

        if isinstance(init_bounds, dict):
            self.init_bounds_list = [init_bounds]
        else:
            self.init_bounds_list = [dict(b) for b in init_bounds]
        self.last_region_idx = 0
        self.goal_xyz = np.asarray(goal_xyz, dtype=np.float64)

        c_inv = max(1.0, 1.0 / self.c)
        self.oracle = CubeMarkovOracle(env=env, min_norm=min_norm,
                                        max_step=int(200 * c_inv))
        self.done = False
        self._xi = 0.0

    def reset_episode(self, rng):
        env = self.env
        region_idx = self.last_region_idx % len(self.init_bounds_list)
        self.last_region_idx = region_idx + 1
        bounds = self.init_bounds_list[region_idx]

        x = float(rng.uniform(bounds['x_low'], bounds['x_high']))
        y = float(rng.uniform(bounds['y_low'], bounds['y_high']))
        yaw = float(rng.uniform(0.0, 2 * np.pi))
        env._data.joint('object_joint_0').qpos[:3] = [x, y, 0.02]
        env._data.joint('object_joint_0').qpos[3:] = lie.SO3.from_z_radians(yaw).wxyz.tolist()

        env._target_task = 'cube'
        env._target_block = 0
        env._data.mocap_pos[env._cube_target_mocap_ids[0]] = self.goal_xyz.copy()
        env._data.mocap_quat[env._cube_target_mocap_ids[0]] = lie.SO3.identity().wxyz.tolist()
        for i in range(1, len(env._cube_target_mocap_ids)):
            env._data.mocap_pos[env._cube_target_mocap_ids[i]] = (0.0, 0.0, -0.3)
            env._data.mocap_quat[env._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()

        mujoco.mj_forward(env._model, env._data)
        info = env.compute_ob_info()
        self.oracle.reset(None, info)
        self.done = False
        self._xi = float(rng.uniform(0.0, self.action_noise)) if self.action_noise > 0 else 0.0

    def get_action(self):
        if self.done:
            return np.zeros(5, dtype=np.float32)
        info = self.env.compute_ob_info()
        action = np.array(self.oracle.select_action(None, info), dtype=np.float64)
        if self._xi > 0.0:
            xi = self._xi
            action = action + np.random.normal(0.0, [xi, xi, xi, xi * 3.0, xi * 10.0])
        action = np.clip(action, -1.0, 1.0)
        action *= self.c
        if self.oracle.done:
            self.done = True
        return action.astype(np.float32)


# ---------------------------------------------------------------------------
# Post-success scripted motion (deterministic; fills H*stride frame budget so
# diffusion windows anchored near success see in-distribution actions).
# ---------------------------------------------------------------------------
# Phase A (t < PHASE_A_STEPS):  PD glide to HOME pose, gripper open
# Phase B (PHASE_A_STEPS ≤ t < POST_SUCCESS_FRAMES):
#                               small circular oscillation around HOME
# Phase C (t ≥ POST_SUCCESS_FRAMES):
#                               zeros (no valid (anchor,window) reaches here)

HOME_POSE = np.array([0.425, -0.15, 0.40], dtype=np.float64)  # opposite side from goal, lifted 40cm
HOME_YAW = 0.0
PHASE_A_STEPS = 30
PHASE_B_PERIOD = 20            # steps per oscillation cycle
PHASE_B_AMP = 0.015            # 1.5 cm radius
POST_SUCCESS_FRAMES = 100      # ≥ max(H*stride) across ablation: 18·5=90, 28·3=84, 42·2=84, 83·1=83
ORACLE_GAIN_POS = 5.0          # match cube_markov.py (in-distribution)
ORACLE_GAIN_YAW = 3.0


def post_success_action(env, t, c):
    """Deterministic scripted action for steps after success_reached.

    All actions use the same PD + gain form as the oracle (ogbench cube_markov),
    so post-success samples lie on the same action manifold the diffusion model
    already trains on for pre-success motion. Gripper held OPEN (-1).
    """
    if t >= POST_SUCCESS_FRAMES:
        return np.zeros(5, dtype=np.float32)

    info = env.compute_ob_info()
    eff_pos = np.asarray(info['proprio/effector_pos'], dtype=np.float64)
    eff_yaw = float(np.asarray(info['proprio/effector_yaw']).flatten()[0])

    if t < PHASE_A_STEPS:
        target_pos = HOME_POSE
    else:
        phase = (t - PHASE_A_STEPS) * 2.0 * np.pi / PHASE_B_PERIOD
        target_pos = HOME_POSE + PHASE_B_AMP * np.array(
            [np.cos(phase), np.sin(phase), 0.0], dtype=np.float64
        )

    diff = target_pos - eff_pos
    action = np.zeros(5, dtype=np.float64)
    action[:3] = diff * ORACLE_GAIN_POS
    action[3] = (HOME_YAW - eff_yaw) * ORACLE_GAIN_YAW
    action[4] = -1.0  # gripper OPEN (oracle convention)
    action = np.clip(action, -1.0, 1.0)
    action *= c
    return action.astype(np.float32)


def check_physical_success(env, goal_xyz, xy_tol=0.04, z_tol=0.04,
                            eff_lift_z=0.15, max_grip_opening=0.4):
    """True only when the expert has actually performed an intentional release:
       (1) cube placed at goal, (2) finger opening commanded open, (3) effector
       lifted above the cube. Slip-based pseudo-releases (cube falls out of a
       still-closed gripper while the arm drifts) do NOT satisfy (2)+(3).
    """
    cube_pos = env._data.joint('object_joint_0').qpos[:3].copy()
    d_xy = float(np.linalg.norm(cube_pos[:2] - np.asarray(goal_xyz)[:2]))
    d_z = abs(float(cube_pos[2] - goal_xyz[2]))
    info = env.compute_ob_info()
    eff_pos = np.asarray(info['proprio/effector_pos'])
    opening = float(np.asarray(info['proprio/gripper_opening']).flatten()[0])
    # Robotiq 2F-85: opening = qpos/0.8, 1=closed 0=open. Plateau on cube
    # is ~0.55-0.6; <0.4 = clearly released past the cube width.
    cube_placed = (d_xy <= xy_tol) and (d_z <= z_tol)
    grip_open   = opening < max_grip_opening
    eff_lifted  = float(eff_pos[2]) >= eff_lift_z
    return cube_placed and grip_open and eff_lifted


# ---------------------------------------------------------------------------
# HDF5 writer (from collect_diffusion_cube.py, with mocap fields added)
# ---------------------------------------------------------------------------

def _get_step_record(env, action, reward=0.0, terminated=False, truncated=False):
    import time as _time
    info = env.compute_ob_info()
    t0 = _time.time()
    info['pixels'] = env.render()
    info['render_time'] = _time.time() - t0
    info['observation'] = env.compute_observation()
    target = getattr(env, '_cur_goal_ob', None)
    if target is None:
        target = np.zeros_like(info['observation'])
    info['target'] = target
    info['action'] = action.astype(np.float32)
    info['step_idx'] = np.int64(0)
    info['success'] = env._success
    info['reward'] = np.float64(reward)
    info['terminated'] = bool(terminated)
    info['truncated'] = bool(truncated)
    # For Phase 2 wrist rendering:
    info['mocap_pos'] = env._data.mocap_pos.copy()
    info['mocap_quat'] = env._data.mocap_quat.copy()
    return info


class H5Writer:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(str(self.path), 'w', libver='latest',
                           fs_strategy='page', fs_page_size=4 * 1024 * 1024)
        self._initialized = False
        self._global_step = 0
        self._ep_count = 0

    def _init_datasets(self, sample_step):
        for key, val in sample_step.items():
            if key.startswith('_'):
                continue
            h5key = key.replace('/', '_')
            arr = np.array(val)
            shape = (0,) + arr.shape
            maxshape = (None,) + arr.shape
            dtype = arr.dtype
            if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.bytes_) or dtype == object:
                dtype = h5py.string_dtype()
            if arr.ndim >= 2:
                chunks = (100,) + arr.shape
                compression = hdf5plugin.Blosc(cname='lz4', clevel=5,
                                               shuffle=hdf5plugin.Blosc.SHUFFLE)
            else:
                chunks = (1000,) + arr.shape
                compression = None
            self.f.create_dataset(h5key, shape=shape, maxshape=maxshape,
                                  dtype=dtype, chunks=chunks, compression=compression)
        self.f.create_dataset('ep_idx', shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(1000,))
        self.f.create_dataset('ep_len', shape=(0,), maxshape=(None,), dtype=np.int32)
        self.f.create_dataset('ep_offset', shape=(0,), maxshape=(None,), dtype=np.int64)
        self._initialized = True

    def write_episode(self, ep_data):
        if not self._initialized:
            self._init_datasets(ep_data[0])
        ep_len = len(ep_data)
        ep_arrays = defaultdict(list)
        for step in ep_data:
            for key, val in step.items():
                if key.startswith('_'):
                    continue
                ep_arrays[key.replace('/', '_')].append(val)
        for h5key, vals in ep_arrays.items():
            ds = self.f[h5key]
            old_len = ds.shape[0]
            ds.resize(old_len + ep_len, axis=0)
            ds[old_len:old_len + ep_len] = np.array(vals)
        ds = self.f['ep_idx']
        old_len = ds.shape[0]
        ds.resize(old_len + ep_len, axis=0)
        ds[old_len:old_len + ep_len] = self._ep_count
        meta_idx = self.f['ep_len'].shape[0]
        self.f['ep_len'].resize(meta_idx + 1, axis=0)
        self.f['ep_offset'].resize(meta_idx + 1, axis=0)
        self.f['ep_len'][meta_idx] = ep_len
        self.f['ep_offset'][meta_idx] = self._global_step
        self._global_step += ep_len
        self._ep_count += 1
        self.f.flush()

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------
# Phase 1: Collect (random init + fixed goal, ends on success)
# ---------------------------------------------------------------------------

def collect_phase1(cfg, name, num_episodes, cache_dir):
    env = gym.make(
        'swm/OGBCube-v0',
        mode='data_collection',
        render_mode='rgb_array',
        width=cfg.world.image_shape[0],
        height=cfg.world.image_shape[1],
        max_episode_steps=cfg.world.max_episode_steps,
        terminate_at_goal=False,
        visualize_info=False,
    )

    init_bounds = dict(cfg.init_bounds)
    goal_xyz = tuple(cfg.goal_xyz)
    logging.info(f'Init bounds: {init_bounds}, Goal: {goal_xyz}')

    policy = FixedGoalCubePolicy(
        env.unwrapped,
        init_bounds=init_bounds,
        goal_xyz=goal_xyz,
        slowdown_factor=cfg.slowdown_factor,
        action_noise=cfg.action_noise,
    )

    rng = np.random.default_rng(cfg.seed)
    max_steps = cfg.world.max_episode_steps
    randomize_arm_init = bool(cfg.get('randomize_arm_init', False))
    # Variable-length episodes: after success_reached, run scripted post-success
    # motion (post_success_action: PD glide to HOME → small oscillation) for
    # exactly POST_SUCCESS_FRAMES steps, then end. Guarantees every anchor's
    # H*stride future window is in-distribution without zero-padding.
    # Override via `+post_success_frames=N`. Set 0 to stop episode at the
    # success step (used by GT-init ablation where ep_len ratio must match c).
    post_success_frames = int(cfg.get('post_success_frames', POST_SUCCESS_FRAMES))
    logging.info(f'Variable-length mode: post_success_frames={post_success_frames}')
    if cfg.get('target_ep_len', None) is not None:
        logging.warning('cfg.target_ep_len is deprecated and ignored. Using variable-length episodes.')

    h5_path = Path(cache_dir) / f'{name}.h5'
    writer = H5Writer(h5_path)

    ep_lens = []
    successes = []

    for ep in tqdm(range(num_episodes), desc='Collecting diffusion cube'):
        ep_seed = rng.integers(0, 1_000_000).item()
        np.random.seed(ep_seed)
        ep_rng = np.random.default_rng(ep_seed)
        # Override variation_space.ee_start_position per-episode. MUST pass via
        # env.reset(options={'variation_values': ...}) — cube_env.reset() calls
        # reset_variation_space() which RESETS variation_space.value to
        # init_value (workspace center) BEFORE applying our value. spaces.py
        # applies variation_values AFTER the reset, so this is the only path
        # that survives. set_value() before env.reset() does NOT work.
        reset_options = {}
        if randomize_arm_init:
            ab = env.unwrapped._arm_sampling_bounds
            eff_pos_random = ep_rng.uniform(ab[0], ab[1])
            reset_options['variation_values'] = {
                'agent.ee_start_position': eff_pos_random,
            }
        env.reset(seed=ep_seed, options=reset_options)
        policy.reset_episode(ep_rng)

        init_action = np.full(5, np.nan, dtype=np.float32)
        ep_data = [_get_step_record(env.unwrapped, init_action)]

        success_reached = False
        post_success_step = 0
        for step in range(max_steps):
            # Stop at success step when post_success_frames==0 (GT-init ablation
            # mode where ep_len must scale with c; no retreat/hover motion).
            if success_reached and post_success_frames <= 0:
                break
            if success_reached:
                # Scripted post-success: Phase A (PD glide to HOME) → Phase B
                # (small oscillation). Generates non-zero in-distribution
                # actions for exactly post_success_frames steps.
                action = post_success_action(env.unwrapped, post_success_step, policy.c)
                post_success_step += 1
            else:
                action = policy.get_action()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_data.append(_get_step_record(env.unwrapped, action,
                                             reward=reward, terminated=terminated,
                                             truncated=truncated))
            if not success_reached and check_physical_success(env.unwrapped, goal_xyz):
                success_reached = True
            # Termination: success ep ends after post_success_frames of scripted
            # motion; failure ep ends when oracle gives up (policy.done).
            if success_reached and post_success_step >= post_success_frames:
                break
            if not success_reached and policy.done:
                break
            if truncated:
                break

        # Align actions (o_t, a_t)
        for i in range(len(ep_data) - 1):
            ep_data[i]['action'] = ep_data[i + 1]['action']
        ep_data[-1]['action'] = np.full(5, np.nan, dtype=np.float32)
        for i, d in enumerate(ep_data):
            d['step_idx'] = np.int64(i)

        # Store success_reached (mid-rollout trigger) instead of final-frame
        # eval — final frame can drift back to fail after padding zero-action.
        ep_lens.append(len(ep_data))
        successes.append(bool(success_reached))
        writer.write_episode(ep_data)

    writer.f.create_dataset('ep_physical_success',
                             data=np.array(successes, dtype=bool))
    writer.close()
    env.close()

    ep_lens = np.array(ep_lens)
    successes = np.array(successes)
    logging.info(f'Phase 1 done: {len(ep_lens)} eps, success={successes.mean():.1%}, '
                 f'avg_len={ep_lens.mean():.1f}')
    return ep_lens, successes


# ---------------------------------------------------------------------------
# Phase 2: Replay + wrist render (chunked + resumable, same as WM scripts)
# ---------------------------------------------------------------------------

def add_wrist_pixels(h5_path, img_size=224, chunk_size=500):
    logging.info(f'Phase 2: Adding wrist_pixels to {h5_path}')

    f = h5py.File(str(h5_path), 'r')
    total_steps = f['qpos'].shape[0]
    has_mocap_pos = 'mocap_pos' in f
    has_mocap_quat = 'mocap_quat' in f
    f.close()

    start_idx = 0
    with h5py.File(str(h5_path), 'r') as f:
        if 'wrist_pixels' in f and '_wrist_progress' in f['wrist_pixels'].attrs:
            start_idx = int(f['wrist_pixels'].attrs['_wrist_progress'])
            if start_idx >= total_steps:
                logging.info(f'Phase 2 already complete ({start_idx}/{total_steps})')
                return
            logging.info(f'Phase 2 RESUMING from step {start_idx}/{total_steps}')

    env = gym.make(
        'swm/OGBCube-v0',
        mode='data_collection',
        render_mode='rgb_array',
        width=img_size, height=img_size,
        max_episode_steps=max(total_steps, 100),
        terminate_at_goal=False,
        visualize_info=False,
    )
    env.reset(seed=0)

    wrist_renderer = WristCameraRenderer(env.unwrapped._model,
                                          height=img_size, width=img_size)
    logging.info(f'Wrist renderer created (site_id={wrist_renderer._site_id})')

    f = h5py.File(str(h5_path), 'a')
    if 'wrist_pixels' not in f:
        f.create_dataset(
            'wrist_pixels',
            shape=(total_steps, img_size, img_size, 3),
            dtype=np.uint8,
            chunks=(100, img_size, img_size, 3),
            compression=hdf5plugin.Blosc(cname='lz4', clevel=5,
                                          shuffle=hdf5plugin.Blosc.SHUFFLE),
        )
        f['wrist_pixels'].attrs['_wrist_progress'] = 0
        f.flush()
    wrist_ds = f['wrist_pixels']

    buf = np.empty((chunk_size, img_size, img_size, 3), dtype=np.uint8)

    with h5py.File(str(h5_path), 'r') as f_read:
        qpos_ds = f_read['qpos']
        qvel_ds = f_read['qvel']
        mocap_pos_ds = f_read['mocap_pos'] if has_mocap_pos else None
        mocap_quat_ds = f_read['mocap_quat'] if has_mocap_quat else None

        pbar = tqdm(total=total_steps, initial=start_idx, desc='Rendering wrist')
        i = start_idx
        while i < total_steps:
            end = min(i + chunk_size, total_steps)
            n = end - i
            qpos_chunk = qpos_ds[i:end]
            qvel_chunk = qvel_ds[i:end]
            mocap_pos_chunk = mocap_pos_ds[i:end] if mocap_pos_ds is not None else None
            mocap_quat_chunk = mocap_quat_ds[i:end] if mocap_quat_ds is not None else None

            for k in range(n):
                env.unwrapped._data.qpos[:] = qpos_chunk[k]
                env.unwrapped._data.qvel[:] = qvel_chunk[k]
                if mocap_pos_chunk is not None:
                    env.unwrapped._data.mocap_pos[:] = mocap_pos_chunk[k]
                if mocap_quat_chunk is not None:
                    env.unwrapped._data.mocap_quat[:] = mocap_quat_chunk[k]
                mujoco.mj_forward(env.unwrapped._model, env.unwrapped._data)
                buf[k] = wrist_renderer.render(env.unwrapped._data)

            wrist_ds[i:end] = buf[:n]
            wrist_ds.attrs['_wrist_progress'] = end
            f.flush()
            i = end
            pbar.update(n)
        pbar.close()

    wrist_renderer.close()
    env.close()
    f.close()
    logging.info(f'Phase 2 done: wrist_pixels ({total_steps} frames) in {h5_path}')


# ---------------------------------------------------------------------------
# Validation (small run + side-by-side videos)
# ---------------------------------------------------------------------------

def validate(cfg):
    cache_dir = cfg.cache_dir
    n_val = cfg.get('n_val_episodes', 5)
    img_size = cfg.get('wrist_cam_img_size', 224)
    # SLOWDOWN-aware so c=1.0 and c=0.3 validates don't collide.
    name = f"cube_multiview_diffusion_val_c{cfg.slowdown_factor}"

    collect_phase1(cfg, name, n_val, cache_dir)
    h5_path = Path(cache_dir) / f'{name}.h5'
    add_wrist_pixels(h5_path, img_size=img_size)

    video_dir = Path(cache_dir) / f'cube_diffusion_validation_videos_c{cfg.slowdown_factor}'
    video_dir.mkdir(parents=True, exist_ok=True)

    f = h5py.File(str(h5_path), 'r')
    ep_len = f['ep_len'][:]
    ep_offset = f['ep_offset'][:]
    successes = f['ep_physical_success'][:]
    pixels = f['pixels']
    wrist_pixels = f['wrist_pixels']

    for ep in range(len(ep_len)):
        off = int(ep_offset[ep])
        L = int(ep_len[ep])
        tag = 'SUCCESS' if successes[ep] else 'FAIL'

        video_path = video_dir / f'episode_{ep:03d}_{tag}.mp4'
        writer = imageio.get_writer(str(video_path), fps=30, codec='libx264',
                                     macro_block_size=1)
        for t in range(L):
            fa = np.array(pixels[off + t])
            fw = np.array(wrist_pixels[off + t])
            sep = np.ones((fa.shape[0], 4, 3), dtype=np.uint8) * 255
            combined = np.hstack([fa, sep, fw])
            writer.append_data(combined)
        writer.close()
        logging.info(f'  ep {ep}: {L} frames, {tag} -> {video_path}')

    f.close()
    logging.info(f'Validation videos saved to {video_dir}')


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path='./config', config_name='multiview_diffusion_cube')
def run(cfg: DictConfig):
    mode = cfg.get('mode', 'collect')
    cache_dir = cfg.cache_dir
    img_size = cfg.get('wrist_cam_img_size', 224)

    if mode == 'validate':
        validate(cfg)
    elif mode == 'collect':
        # Optional name suffix to distinguish runs (e.g., _c0.3_afterclip).
        # `+name=...` from submit_collect script takes precedence; legacy
        # `name_suffix` is fallback.
        name = cfg.get('name',
                       'cube_multiview_diffusion' + str(cfg.get('name_suffix', '')))
        collect_phase1(cfg, name, cfg.num_traj, cache_dir)
        h5_path = Path(cache_dir) / f'{name}.h5'
        add_wrist_pixels(h5_path, img_size=img_size)
        logging.success(f'Done: {name} ({cfg.num_traj} episodes)')
    else:
        raise ValueError(f'Unknown mode: {mode}')


if __name__ == '__main__':
    run()
