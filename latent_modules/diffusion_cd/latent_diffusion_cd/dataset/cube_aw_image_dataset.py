"""CleanDiffuser dataset adapter for OGBench cube AW multiview H5.

Bridges our flat-layout H5 (with ep_offset / ep_len) to CleanDiffuser's
zarr-MemoryStore + Jpeg2k ReplayBuffer expected by ChiUNet1d / DiT1d image DP.

Source H5 layout (cube_multiview_diffusion_c0.3_postpad_fullscale.h5):
    pixels         (N_total, 224, 224, 3) uint8       # agentview
    wrist_pixels   (N_total, 224, 224, 3) uint8       # wrist
    action         (N_total, 5)            float32    # xyz + yaw + gripper
    ep_offset      (n_eps,)                int64
    ep_len         (n_eps,)                int32

We resize 224 -> 96 on load (lazy, per-frame); CropRandomizer in
MultiImageObsCondition handles 96 -> 76 train / center 76 eval.
"""
from typing import Dict
import h5py
try:
    import hdf5plugin  # noqa: F401 — registers BLOSC/LZF/etc. plugins for compressed h5
except ImportError:
    pass
import zarr
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cleandiffuser.dataset.imagecodecs import register_codecs, Jpeg2k
from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.replay_buffer import ReplayBuffer
from cleandiffuser.dataset.dataset_utils import (
    SequenceSampler, MinMaxNormalizer, ImageNormalizer, dict_apply,
)

register_codecs()


def _resize_uint8_chw(arr_hwc_uint8: np.ndarray, target: int) -> np.ndarray:
    """Bilinear resize (T, H, W, C) uint8 -> (T, target, target, C) uint8."""
    if arr_hwc_uint8.shape[1] == target and arr_hwc_uint8.shape[2] == target:
        return arr_hwc_uint8
    x = torch.from_numpy(arr_hwc_uint8).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
    x = x.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
    return x


def _convert_cube_aw_to_replay(store, shape_meta, dataset_path, max_episodes=None):
    """Read our flat H5, resize images, write to zarr ReplayBuffer.

    Mirrors `_convert_robomimic_to_replay` minus the rotation transformer
    and the per-demo group iteration (we iterate via ep_offset/ep_len).
    ``max_episodes`` (default None = all) truncates to a smoke-friendly subset.
    """
    obs_shape_meta = shape_meta["obs"]
    rgb_keys = [k for k, a in obs_shape_meta.items() if a.get("type") == "rgb"]

    # Map shape_meta keys to source-H5 keys.
    src_key_map = {
        "agentview_image": "pixels",
        "wrist_image": "wrist_pixels",
    }
    for k in rgb_keys:
        if k not in src_key_map:
            raise ValueError(
                f"shape_meta has rgb key {k!r} but no source mapping; "
                f"known mappings: {list(src_key_map)}"
            )

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    with h5py.File(dataset_path, "r") as f:
        ep_offset = f["ep_offset"][:].astype(np.int64)
        ep_len = f["ep_len"][:].astype(np.int64)
        if max_episodes is not None:
            ep_offset = ep_offset[:int(max_episodes)]
            ep_len = ep_len[:int(max_episodes)]
            print(f"[cube_aw] max_episodes={max_episodes} → truncated")
        n_eps = len(ep_offset)
        episode_ends = np.cumsum(ep_len).astype(np.int64)
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        n_steps = int(episode_ends[-1])
        print(f"[cube_aw] n_eps={n_eps}  n_steps={n_steps}  "
              f"avg_ep_len={n_steps/max(n_eps,1):.1f}")

        meta_group.array("episode_ends", episode_ends,
                         dtype=np.int64, compressor=None, overwrite=True)

        # ---- lowdim: action (no obs lowdim keys for cube AW pixel-DP) ----
        # nan_to_num: H5 padding zones can have NaN actions (post-success freeze
        # frames). MinMaxNormalizer would propagate NaN through min/max.
        action_src = np.empty((n_steps,) + tuple(shape_meta["action"]["shape"]),
                              dtype=np.float32)
        for i in tqdm(range(n_eps), desc="[cube_aw] loading actions"):
            lo, ln = int(ep_offset[i]), int(ep_len[i])
            raw_a = f["action"][lo:lo + ln].astype(np.float32)
            raw_a = np.nan_to_num(raw_a, nan=0.0)
            action_src[episode_starts[i]:episode_starts[i] + ln] = raw_a
        data_group.array(
            name="action", data=action_src,
            shape=action_src.shape, chunks=action_src.shape,
            compressor=None, dtype=np.float32,
        )

        # ---- rgb: agentview_image + wrist_image, resized + Jpeg2k ----
        for key in rgb_keys:
            shape = tuple(obs_shape_meta[key]["shape"])    # (3, H, W) CHW
            c, h, w = shape
            assert c == 3, f"expected 3-channel rgb, got {shape}"
            src_h5_key = src_key_map[key]
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),       # HWC for zarr
                chunks=(1, h, w, c),
                compressor=Jpeg2k(level=50),
                dtype=np.uint8,
            )
            # Per-episode resize+write to keep RAM bounded (one episode worth at a time).
            for i in tqdm(range(n_eps), desc=f"[cube_aw] loading {key}"):
                lo, ln = int(ep_offset[i]), int(ep_len[i])
                raw = f[src_h5_key][lo:lo + ln]            # (ln, 224, 224, 3) uint8
                resized = _resize_uint8_chw(raw, h)        # (ln, h, w, 3) uint8
                start = int(episode_starts[i])
                # Per-frame write (zarr chunk == 1 frame, Jpeg2k compress per-chunk).
                for t in range(ln):
                    img_arr[start + t] = resized[t]

    return ReplayBuffer(root)


class CubeAWImageDataset(BaseDataset):
    """OGBench cube AW multiview H5 -> CleanDiffuser image-DP format.

    Drop-in replacement for `RobomimicImageDataset`. Differences:
      - Our action is 5-d (xyz + yaw + gripper); no rotation transformer.
      - Source H5 is flat with ep_offset/ep_len, not per-demo groups.
      - Resizes 224 -> shape_meta image_size (96 default) on load.
    """

    def __init__(self,
                 dataset_dir,
                 shape_meta: dict,
                 n_obs_steps: int = 2,
                 horizon: int = 16,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 max_episodes: int = None):
        super().__init__()
        self.replay_buffer = _convert_cube_aw_to_replay(
            store=zarr.MemoryStore(),
            shape_meta=shape_meta,
            dataset_path=dataset_dir,
            max_episodes=max_episodes,
        )

        rgb_keys, lowdim_keys = [], []
        for k, a in shape_meta["obs"].items():
            t = a.get("type", "low_dim")
            (rgb_keys if t == "rgb" else lowdim_keys).append(k)

        key_first_k = {}
        if n_obs_steps is not None:
            for k in rgb_keys + lowdim_keys:
                key_first_k[k] = n_obs_steps
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            key_first_k=key_first_k,
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.normalizer = self.get_normalizer()

    def get_normalizer(self):
        from collections import defaultdict
        normalizer = defaultdict(dict)
        for k in self.lowdim_keys:
            normalizer["obs"][k] = MinMaxNormalizer(self.replay_buffer[k][:])
        for k in self.rgb_keys:
            normalizer["obs"][k] = ImageNormalizer()
        normalizer["action"] = MinMaxNormalizer(self.replay_buffer["action"][:])
        return normalizer

    def undo_transform_action(self, action):
        """No rotation transform; cube AW yaw is a scalar passed through."""
        return action

    def __str__(self) -> str:
        return (f"CubeAWImageDataset(keys={self.replay_buffer.keys()} "
                f"steps={self.replay_buffer.n_steps} "
                f"episodes={self.replay_buffer.n_episodes})")

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        obs_dict = {}
        # rgb: uint8 (T, H, W, C) -> float32 /255 in [0,1] -> ImageNormalizer
        # maps to [-1, +1] -> reshape to (T, C, H, W) for MultiImageObsCondition.
        T_obs = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
        for k in self.rgb_keys:
            img = sample[k][:T_obs].astype(np.float32) / 255.0     # (T_obs, H, W, C) in [0,1]
            img = self.normalizer["obs"][k].normalize(img)          # -> [-1, +1]
            img = np.moveaxis(img, -1, -3).astype(np.float32)       # (T_obs, C, H, W)
            obs_dict[k] = img
        for k in self.lowdim_keys:
            v = sample[k][:T_obs].astype(np.float32)
            v = self.normalizer["obs"][k].normalize(v).astype(np.float32)
            obs_dict[k] = v

        action = sample["action"].astype(np.float32)                # (horizon, 5)
        action = np.nan_to_num(action, nan=0.0)
        action = self.normalizer["action"].normalize(action).astype(np.float32)

        data = {"obs": obs_dict, "action": action}
        return dict_apply(data, torch.tensor)
