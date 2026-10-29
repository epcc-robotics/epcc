"""Collect multiview world-model data for OGBench Cube.

Two-phase approach:
  1. Collect using official swm.World + ExpertPolicy (identical to original dataset)
  2. Replay qpos/qvel to render wrist camera and append wrist_pixels to HDF5

This guarantees actions/trajectories match the original exactly.

Usage:
    # Validation: 5 episodes + side-by-side videos
    python collect_multiview_WM_cube.py mode=validate

    # Full collection
    python collect_multiview_WM_cube.py mode=collect
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

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

import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench import ExpertPolicy


class SpeedScaledExpertPolicy(ExpertPolicy):
    """Afterclip pace knob for play collection (D-grid).

    The parent ExpertPolicy clips the oracle action to [-1, 1] as its last step
    (swm_srl_v2 expert_policy.py:272); we scale the RETURNED array by c — i.e.
    clip -> *c, all 5 dims, exactly the ordering the c=0.3 demo corpus was
    produced with (le-wm collect_multiview_diffusion_cube.py working tree).
    c=1.0 is a bit-exact no-op. Oracle done/goal-resample logic runs on the
    unscaled action inside the parent and is untouched.
    """

    def __init__(self, *args, slowdown_factor: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = float(slowdown_factor)

    def get_action(self, *args, **kwargs):
        actions = super().get_action(*args, **kwargs)
        if self.c != 1.0:
            actions = (np.asarray(actions) * self.c).astype(np.float32)
        return actions


# ---------------------------------------------------------------------------
# Wrist camera renderer
# ---------------------------------------------------------------------------

class WristCameraRenderer:
    """True first-person eye-in-hand via direct scene.camera override.

    Bypasses MjvCamera spherical coords entirely. Sets scene.camera[0].pos/
    forward/up directly after update_scene() — giving full 6DOF control with
    NO gimbal lock. The camera truly rotates with the wrist.
    """

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
            logging.warning('attachment_site not found — wrist cam disabled')

    def render(self, data):
        if self._site_id is None:
            return np.zeros((self.renderer.height, self.renderer.width, 3), dtype=np.uint8)

        ee_pos = data.site_xpos[self._site_id].copy()
        ee_mat = data.site_xmat[self._site_id].reshape(3, 3).copy()

        # Use site's own axes for full 6DOF camera (rotates with the wrist!)
        # +ee_mat[:, 2] = toward fingertips (Robotiq pinch_site at local +Z)
        # +ee_mat[:, 1] = camera "up" reference (site Y axis, perpendicular)
        forward_raw = ee_mat[:, 2]      # toward fingertips / object
        up_raw = ee_mat[:, 1]           # site Y axis

        # Tilt camera aim toward the gripper (add -up component to forward)
        # so the cube ends up in the center of the frame.
        tilt = 0.175  # fraction of up to mix into forward (downward tilt)
        forward = forward_raw - up_raw * tilt
        forward = forward / np.linalg.norm(forward)

        # Recompute up to stay orthogonal to the new forward
        up = up_raw - np.dot(up_raw, forward) * forward
        up = up / np.linalg.norm(up)

        # Camera position: offset sideways (planar) away from the arm body,
        # plus slight backward offset to frame the gripper in the view.
        cam_pos = ee_pos + up_raw * 0.06 - forward_raw * 0.015

        # update_scene populates scene.camera using the MjvCamera.
        # We then OVERRIDE the scene's GL camera fields directly.
        self.renderer.update_scene(data, camera=self.cam, scene_option=self.scene_option)
        scene = self.renderer.scene

        # Set both stereo cameras to our 6DOF transform
        for i in range(2):
            scene.camera[i].pos[:] = cam_pos.astype(np.float32)
            scene.camera[i].forward[:] = forward.astype(np.float32)
            scene.camera[i].up[:] = up.astype(np.float32)

        return self.renderer.render().copy()

    def close(self):
        if hasattr(self, 'renderer'):
            self.renderer.close()


# ---------------------------------------------------------------------------
# Phase 1: Collect with official pipeline
# ---------------------------------------------------------------------------

def _patch_set_new_target(env):
    """Patch set_new_target to force fresh random goal positions.

    The original set_new_target uses variation_space.value which is cached
    from reset(), returning the SAME target every call within an episode.
    This patch samples fresh positions using env.np_random instead.
    """
    from ogbench.manipspace import lie as _lie
    _original_set_new_target = env.set_new_target

    def _patched_set_new_target(return_info=True, p_stack=0.5):
        # Sample fresh goal position (bypass cached variation_space.value)
        xy = env.np_random.uniform(*env._target_sampling_bounds)
        yaw = env.np_random.uniform(-np.pi, np.pi)
        tar_pos = np.array([xy[0], xy[1], 0.02])
        tar_ori = _lie.SO3.from_z_radians(yaw).wxyz.tolist()

        # Set mocap target for cube 0 (single cube env)
        env._target_block = 0
        env._data.mocap_pos[env._cube_target_mocap_ids[0]] = tar_pos
        env._data.mocap_quat[env._cube_target_mocap_ids[0]] = tar_ori

        if return_info:
            return env.compute_observation(), env.get_reset_info()

    env.set_new_target = _patched_set_new_target


def _patch_compute_ob_info(env):
    """Augment compute_ob_info to include mocap state for Phase 2 restoration."""
    _orig = env.compute_ob_info

    def _patched():
        info = _orig()
        info['mocap_pos'] = env._data.mocap_pos.copy()
        info['mocap_quat'] = env._data.mocap_quat.copy()
        return info

    env.compute_ob_info = _patched


def _recorded_h5_path(cache_dir, name):
    """Where swm_srl_v2's record_dataset actually writes."""
    return Path(cache_dir) / 'datasets' / f'{name}.h5'


def collect_official(cfg, name, num_episodes, cache_dir):
    """Collect using swm.World + SpeedScaledExpertPolicy with patched set_new_target."""
    c = float(cfg.get('slowdown_factor', 1.0))

    # Keep the mid-episode goal-resample DENSITY invariant across c: at c<1 the
    # arm moves 1/c slower, so an episode needs ~1/c more steps to hit the same
    # number of oracle-done -> set_new_target events (the defining trait of the
    # play corpus). c>=1 keeps the configured length.
    world_kwargs = dict(cfg.world)
    if c < 1.0:
        base = int(world_kwargs.get('max_episode_steps', 200))
        world_kwargs['max_episode_steps'] = int(round(base / c))
        logging.info(
            f'slowdown_factor={c} < 1: max_episode_steps {base} -> '
            f"{world_kwargs['max_episode_steps']} (resample-density compensation)"
        )

    world = swm.World(
        'swm/OGBCube-v0', **world_kwargs,
        multiview=False, width=224, height=224,
        visualize_info=False, terminate_at_goal=False, mode='data_collection',
    )

    # Patch set_new_target on each sub-env to fix cached variation_space
    # and patch compute_ob_info to save mocap state for Phase 2.
    for env in world.envs.unwrapped.envs:
        _patch_set_new_target(env.unwrapped)
        _patch_compute_ob_info(env.unwrapped)

    policy = SpeedScaledExpertPolicy(
        action_noise=cfg.action_noise, p_random_action=0.0,
        slowdown_factor=c,
    )
    world.set_policy(policy)

    rng = np.random.default_rng(cfg.seed)
    world.record_dataset(
        name, episodes=num_episodes,
        seed=rng.integers(0, 1_000_000).item(),
        cache_dir=cache_dir,
    )

    # Provenance attrs (the swm recorder writes none itself).
    # NOTE: current swm_srl_v2 record_dataset writes under {cache_dir}/datasets/
    # (the Apr-2026 original wrote to {cache_dir} directly).
    h5_path = _recorded_h5_path(cache_dir, name)
    with h5py.File(str(h5_path), 'a') as f:
        f.attrs['slowdown_factor'] = c
        f.attrs['policy_source'] = 'ogbench_markov_oracle_afterclip'
        f.attrs['dataset_kind'] = 'play'
        f.attrs['collect_seed'] = int(cfg.seed)
    logging.info(f'Phase 1 done: {num_episodes} episodes -> {cache_dir}/{name}.h5')


# ---------------------------------------------------------------------------
# Phase 2: Replay and render wrist camera
# ---------------------------------------------------------------------------

def add_wrist_pixels(h5_path, img_size=224, chunk_size=500):
    """Replay qpos/qvel from HDF5 and append wrist_pixels column.

    Chunked + resumable:
    - Writes to HDF5 in chunks (no memory blow-up for large datasets)
    - Tracks progress via a '_wrist_progress' attr — re-running resumes
      from where it left off if the job was killed mid-processing.
    """
    logging.info(f'Phase 2: Adding wrist_pixels to {h5_path}')

    f = h5py.File(str(h5_path), 'r')
    total_steps = f['qpos'].shape[0]
    has_mocap_pos = 'mocap_pos' in f
    has_mocap_quat = 'mocap_quat' in f
    f.close()

    # Check for existing progress (resume support)
    start_idx = 0
    with h5py.File(str(h5_path), 'r') as f:
        if 'wrist_pixels' in f and '_wrist_progress' in f['wrist_pixels'].attrs:
            start_idx = int(f['wrist_pixels'].attrs['_wrist_progress'])
            if start_idx >= total_steps:
                logging.info(f'Phase 2: already complete ({start_idx}/{total_steps}) — skip')
                return
            logging.info(f'Phase 2: RESUMING from step {start_idx}/{total_steps}')

    env = gym.make(
        'swm/OGBCube-v0',
        mode='data_collection',
        render_mode='rgb_array',
        width=img_size, height=img_size,
        max_episode_steps=300,
        terminate_at_goal=False,
        visualize_info=False,
    )
    env.reset(seed=0)

    wrist_renderer = WristCameraRenderer(env.unwrapped._model, height=img_size, width=img_size)
    logging.info(f'Wrist renderer created (site_id={wrist_renderer._site_id})')

    # Open HDF5 in append mode; create dataset with full size if missing
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

    # Process in chunks, flushing after each chunk for crash safety
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
            # Preload chunk into memory
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

            # Write chunk to HDF5 and update progress marker
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
# Validation: collect a few episodes and save side-by-side videos
# ---------------------------------------------------------------------------

def validate(cfg):
    """Collect small dataset, add wrist pixels, save side-by-side videos."""
    cache_dir = cfg.cache_dir
    n_val = cfg.get('n_val_episodes', 5)
    img_size = cfg.get('wrist_cam_img_size', 224)
    name = f"cube_multiview_WM_val_c{cfg.get('slowdown_factor', 1.0)}"

    # Phase 1: collect with official pipeline
    collect_official(cfg, name, n_val, cache_dir)

    # Phase 2: add wrist pixels
    h5_path = _recorded_h5_path(cache_dir, name)
    add_wrist_pixels(h5_path, img_size=img_size)

    # Save side-by-side videos (per-c dir: parallel c-jobs must not collide)
    video_dir = Path(cache_dir) / f"cube_validation_videos_c{cfg.get('slowdown_factor', 1.0)}"
    video_dir.mkdir(parents=True, exist_ok=True)

    f = h5py.File(str(h5_path), 'r')
    ep_len = f['ep_len'][:]
    ep_offset = f['ep_offset'][:]
    pixels = f['pixels']
    wrist_pixels = f['wrist_pixels']

    for ep in range(len(ep_len)):
        off = int(ep_offset[ep])
        L = int(ep_len[ep])

        video_path = video_dir / f'episode_{ep:03d}.mp4'
        writer = imageio.get_writer(str(video_path), fps=30, codec='libx264')
        for t in range(L):
            fa = np.array(pixels[off + t])
            fw = np.array(wrist_pixels[off + t])
            sep = np.ones((fa.shape[0], 4, 3), dtype=np.uint8) * 255
            combined = np.hstack([fa, sep, fw])
            writer.append_data(combined)
        writer.close()
        logging.info(f'  ep {ep}: {L} frames -> {video_path}')

    f.close()
    logging.info(f'Validation videos saved to {video_dir}')
    logging.info('Layout: [agentview (left) | wrist_cam (right)]')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path='./config', config_name='multiview_wm_cube')
def run(cfg: DictConfig):
    mode = cfg.get('mode', 'collect')
    cache_dir = cfg.cache_dir
    img_size = cfg.get('wrist_cam_img_size', 224)

    if mode == 'validate':
        validate(cfg)
    elif mode == 'collect':
        # ALWAYS c-suffixed (even c=1.0): the champion corpus cube_multiview_WM.h5
        # lives in the same cache_dir and record_dataset APPENDS on name collision.
        name = cfg.get('name') or f"cube_multiview_WM_c{cfg.get('slowdown_factor', 1.0)}"
        collect_official(cfg, name, cfg.num_traj, cache_dir)
        h5_path = _recorded_h5_path(cache_dir, name)
        add_wrist_pixels(h5_path, img_size=img_size)
        logging.success(f'Done: {name} ({cfg.num_traj} episodes)')
    else:
        raise ValueError(f'Unknown mode: {mode}')


if __name__ == '__main__':
    run()
