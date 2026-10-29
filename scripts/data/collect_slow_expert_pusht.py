"""Collect slow PushT demos with the HuggingFace lerobot pretrained diffusion policy.

Uses ``lerobot/diffusion_pusht`` as the expert, with action scale ``slowdown_factor < 1`` for slower motion.
No gym_pusht dependency — ``swm/PushT-v1`` only.

Usage:
    python scripts/collect_slow_expert_pusht.py slowdown_factor=0.5 num_traj=50 dataset_name=pusht_dp_c0.5
    python scripts/collect_slow_expert_pusht.py success_only=false num_traj=100
"""

import os

os.environ["MUJOCO_GL"] = "egl"

# Monkey-patch lerobot groot before any lerobot imports (Python 3.10 compat)
import sys
import types

groot_mod = types.ModuleType("lerobot.policies.groot")
groot_mod.configuration_groot = types.ModuleType("lerobot.policies.groot.configuration_groot")
groot_mod.configuration_groot.GrootConfig = type("GrootConfig", (), {})
sys.modules["lerobot.policies.groot"] = groot_mod
sys.modules["lerobot.policies.groot.configuration_groot"] = groot_mod.configuration_groot
sys.modules["lerobot.policies.groot.__init__"] = groot_mod
sys.modules["lerobot.policies.groot.modeling_groot"] = types.ModuleType("lerobot.policies.groot.modeling_groot")
sys.modules["lerobot.policies.groot.modeling_groot"].GrootPolicy = type("GrootPolicy", (), {})

from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import stable_worldmodel as swm
from huggingface_hub import hf_hub_download
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from omegaconf import DictConfig
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def _build_lerobot_dp(lr_cfg, device="cuda"):
    """Build and load the pretrained lerobot diffusion policy from config."""
    config = DiffusionConfig(
        input_features={
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
        n_obs_steps=lr_cfg.n_obs_steps,
        horizon=lr_cfg.horizon,
        n_action_steps=lr_cfg.n_action_steps,
        vision_backbone=lr_cfg.vision_backbone,
        crop_shape=tuple(lr_cfg.crop_shape),
        crop_is_random=False,
        use_group_norm=lr_cfg.use_group_norm,
        spatial_softmax_num_keypoints=lr_cfg.spatial_softmax_num_keypoints,
        down_dims=tuple(lr_cfg.down_dims),
        kernel_size=lr_cfg.kernel_size,
        n_groups=lr_cfg.n_groups,
        diffusion_step_embed_dim=lr_cfg.diffusion_step_embed_dim,
        use_film_scale_modulation=lr_cfg.use_film_scale_modulation,
        num_train_timesteps=lr_cfg.num_train_timesteps,
        beta_schedule=lr_cfg.beta_schedule,
        prediction_type=lr_cfg.prediction_type,
        clip_sample=lr_cfg.clip_sample,
        clip_sample_range=lr_cfg.clip_sample_range,
        num_inference_steps=lr_cfg.num_inference_steps,
    )

    policy = DiffusionPolicy(config)
    weights_path = hf_hub_download(lr_cfg.repo_id, "model.safetensors", revision=lr_cfg.revision)
    state_dict = load_file(weights_path)
    model_keys = {
        k: v
        for k, v in state_dict.items()
        if not k.startswith(("normalize_inputs.", "normalize_targets.", "unnormalize_outputs."))
    }
    policy.load_state_dict(model_keys, strict=False)
    return policy.to(device).eval()


class LeRobotDPPolicy(swm.policy.BasePolicy):
    """Wraps lerobot pretrained DP for use with swm/PushT-v1."""

    def __init__(self, lr_cfg, slowdown_factor=1.0, device="cuda"):
        super().__init__()
        self.slowdown_factor = slowdown_factor
        self.device = torch.device(device)
        self.dp = _build_lerobot_dp(lr_cfg, device=device)
        self.dp.requires_grad_(False)

        # Load normalization stats from config
        norm = lr_cfg.norm
        self._img_mean = norm.img_mean
        self._img_std = norm.img_std
        self._state_min = np.array(norm.state_min, dtype=np.float32)
        self._state_max = np.array(norm.state_max, dtype=np.float32)
        self._action_min = np.array(norm.action_min, dtype=np.float32)
        self._action_max = np.array(norm.action_max, dtype=np.float32)
        self._action_scale = lr_cfg.action_scale

    def _preprocess_image(self, pixels):
        """(n_envs, H, W, 3) uint8 -> (n_envs, 3, 96, 96) float [-1,1]."""
        img = torch.from_numpy(pixels).float().to(self.device) / 255.0
        img = img.permute(0, 3, 1, 2)
        img = F.interpolate(img, size=(96, 96), mode="bilinear", align_corners=False)
        img = (img - self._img_mean) / self._img_std
        return img

    def _normalize_state(self, agent_pos):
        """(n_envs, 2) in [0,512] -> [-1,1]."""
        return (agent_pos - self._state_min) / (self._state_max - self._state_min) * 2.0 - 1.0

    def _denormalize_action(self, action_norm):
        """(n_envs, 2) in [-1,1] -> absolute pixel coords."""
        return (action_norm + 1.0) / 2.0 * (self._action_max - self._action_min) + self._action_min

    def get_action(self, info_dict, **kwargs):
        pixels = info_dict["pixels"][:, -1]
        agent_pos = np.array(info_dict["state"][:, -1, :2], dtype=np.float32)

        n_envs = pixels.shape[0]
        actions_out = np.zeros((n_envs, 2), dtype=np.float32)

        for i in range(n_envs):
            img_t = self._preprocess_image(pixels[i : i + 1])
            state_norm = self._normalize_state(agent_pos[i : i + 1])
            state_t = torch.from_numpy(state_norm).float().to(self.device)

            obs_dict = {
                "observation.image": img_t,
                "observation.state": state_t,
            }

            with torch.no_grad():
                action_norm = self.dp.select_action(obs_dict)
                action_norm = action_norm.squeeze(0).cpu().numpy()

            abs_target = self._denormalize_action(action_norm)
            displacement = abs_target - agent_pos[i]
            relative = displacement / self._action_scale * self.slowdown_factor

            actions_out[i] = np.clip(relative, -1.0, 1.0)

        return actions_out

    def set_env(self, env):
        super().set_env(env)
        self.dp.reset()

    def reset(self):
        self.dp.reset()


def patch_pusht_block_only_success(world):
    """Override PushT eval_state to check block vs visual goal_pose only."""
    sync = world.envs.unwrapped
    for env in sync.envs:
        base = env
        while hasattr(base, "env"):
            base = base.env

        def _block_only_eval_state(goal_state, cur_state):
            block_pos_diff = np.linalg.norm(goal_state[2:4] - cur_state[2:4])
            angle_diff = np.abs(goal_state[4] - cur_state[4])
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            success = block_pos_diff < 20 and angle_diff < np.pi / 9
            state_dist = np.linalg.norm(goal_state - cur_state)
            return success, state_dist

        base.eval_state = _block_only_eval_state

        _orig_reset = base.reset

        def _patched_reset(*args, _base=base, _orig=_orig_reset, **kwargs):
            result = _orig(*args, **kwargs)
            _base.goal_state[2] = _base.goal_pose[0]
            _base.goal_state[3] = _base.goal_pose[1]
            _base.goal_state[4] = _base.goal_pose[2]
            return result

        base.reset = _patched_reset


@hydra.main(version_base=None, config_path="config", config_name="expert_pusht")
def run(cfg: DictConfig):
    slowdown_factor = cfg.slowdown_factor
    num_traj = cfg.num_traj
    cache_dir = os.path.join(os.environ.get("STABLEWM_HOME"), "pusht")
    seed = cfg.seed
    device = cfg.device
    lr_cfg = cfg.lerobot
    num_envs = cfg.world.num_envs
    max_steps = cfg.world.max_episode_steps
    dataset_name = cfg.dataset_name

    # world with updated success criterion
    world = swm.World(
        "swm/PushT-v1",
        num_envs=num_envs,
        max_episode_steps=max_steps,
        image_shape=cfg.world.image_shape,
        history_size=1,
        frame_skip=1,
    )
    patch_pusht_block_only_success(world)

    # slow expert policy using le-robot
    policy = LeRobotDPPolicy(lr_cfg=lr_cfg, slowdown_factor=slowdown_factor, device=device)
    world.set_policy(policy)

    print(f"[pusht-dp] Collecting {num_traj} episodes (slowdown_factor={slowdown_factor}) -> {dataset_name}")
    world.record_dataset(dataset_name, episodes=num_traj, seed=seed, cache_dir=cache_dir)
    print(f"[pusht-dp] Saved to {cache_dir}/{dataset_name}.h5")

    # Log success rate and avg time-to-completion
    ds = swm.data.HDF5Dataset(dataset_name, frameskip=1, cache_dir=cache_dir)
    n_eps = len(ds.lengths)
    n_success = 0
    completion_steps = []
    for i in range(n_eps):
        ep = ds.load_episode(i)
        terminated = ep.get("terminated", None)
        if terminated is not None:
            if hasattr(terminated, "numpy"):
                terminated = terminated.numpy()
            if np.any(terminated):
                n_success += 1
                first_t = int(np.argmax(terminated))
                completion_steps.append(first_t)

    sr = 100 * n_success / n_eps if n_eps > 0 else 0
    avg_steps = np.mean(completion_steps) if completion_steps else float("nan")
    print(f"[pusht-dp] Stats: {n_success}/{n_eps} success ({sr:.0f}% SR), "
          f"avg completion steps: {avg_steps:.0f}, "
          f"episode lengths: {ds.lengths.tolist()}")


if __name__ == "__main__":
    run()
