"""Cube AW latent-trajectory dataset for CleanDiffuser.

Port of `latent_diffusion.dataset.CubeLatentDataset` to the cleandiffuser tree,
with the normalizer made pluggable via `make_normalizer(kind, X)`.

Returns per-window diffusion samples:
    obs:    (H, D)          future latent trajectory (the diffusion target)
    cond:   (K*D,)          flat history-only conditioning (K stride-spaced frames)
    is_pad: (H,) bool       True at padded/clamped positions (DP modes only)

History and target overlap by one frame: `history[-1] == target[0]` — matches the
`fix_mask[0]=1` inpainting pattern used by VelLossDiffusionSDE in the 70% baseline.

Constants for our 6-ablation study (cube AW c=0.3 H5):
    DEFAULT_H5: cube_multiview_diffusion_aw_c0.3.h5  (~190K frames × 384-d)
    K=2, H=18, stride=5, padding_mode="valid_only"
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import h5py
try:
    import hdf5plugin  # noqa: F401 — register BLOSC/LZF filters for compressed H5
except ImportError:
    pass
import numpy as np
import torch
from torch.utils.data import Dataset

from latent_diffusion_cd.dataset.normalizers import make_normalizer


DEFAULT_H5 = os.path.join(
    os.environ.get("STABLEWM_HOME", ""),
    "cube/latents/cube_multiview_diffusion_aw_c0.3_postpad_fullscale.h5",
)


def _read_success(f: h5py.File) -> np.ndarray:
    if "ep_success" in f:
        return np.asarray(f["ep_success"][:], dtype=bool)
    if "ep_physical_success" in f:
        return np.asarray(f["ep_physical_success"][:], dtype=bool)
    n = int(f["ep_len"].shape[0])
    return np.ones(n, dtype=bool)


def _read_valid_len(f: h5py.File) -> np.ndarray:
    if "valid_len" in f:
        return np.asarray(f["valid_len"][:], dtype=np.int64)
    return np.asarray(f["ep_len"][:], dtype=np.int64)


class CubeLatentDataset(Dataset):
    """Windowed reader over cached latent H5. Pluggable normalizer.

    Args:
        h5_path: absolute path to cube_multiview_diffusion_aw_c0.3*.h5
        horizon: H, latent target window length (default 18 = 70% baseline)
        history: K, history frames count (default 2)
        stride: env-frames between adjacent latent frames (default 5)
        split: "train" | "val"
        val_frac: fraction of successful episodes held out for val
        max_episodes: cap on episode count for fast smokes
        seed: train/val deterministic split seed
        normalizer_kind: "standard" | "minmax" | "none" — passed to make_normalizer
        padding_mode: "valid_only" | "dp" | "half_dp" — controls anchor range +
            last-frame clamp behavior
    """
    def __init__(
        self,
        h5_path: str = DEFAULT_H5,
        horizon: int = 18,
        history: int = 2,
        stride: int = 5,
        split: str = "train",
        val_frac: float = 0.1,
        max_episodes: Optional[int] = None,
        seed: int = 42,
        normalizer_kind: str = "standard",
        padding_mode: str = "valid_only",
    ):
        assert split in ("train", "val")
        assert padding_mode in ("valid_only", "dp", "half_dp"), padding_mode
        self.h5_path = h5_path
        self.H = int(horizon)
        self.K = int(history)
        self.stride = int(stride)
        self.max_episodes = int(max_episodes) if max_episodes else -1
        self.val_frac = float(val_frac)
        self.padding_mode = padding_mode
        self.seed = int(seed)
        self.normalizer_kind = str(normalizer_kind)
        assert self.H > 0 and self.K >= 1 and self.stride >= 1

        # ---- episode metadata + latent shape ----
        with h5py.File(h5_path, "r") as f:
            ep_success = _read_success(f)
            ep_offset = f["ep_offset"][:]
            valid_len = _read_valid_len(f)
            latent_shape = f["latent"].shape
        self.latent_dim = int(latent_shape[1])

        # ---- pick successful eps + cap ----
        success_eps = np.where(ep_success)[0]
        if max_episodes and max_episodes > 0:
            success_eps = success_eps[: int(max_episodes)]
        assert len(success_eps) > 0, "no successful episodes after filtering"

        # ---- train/val split by episode id (deterministic) ----
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(success_eps))
        shuffled = success_eps[perm]
        n_val = 0 if val_frac <= 0 else max(1, int(round(len(shuffled) * val_frac)))  # val_frac<=0 -> no val (tiny-N)
        val_eps = np.sort(shuffled[:n_val])
        train_eps = np.sort(shuffled[n_val:])
        self.episodes = train_eps if split == "train" else val_eps

        # ---- index list of (offset, t, vl) tuples ----
        idx_list = []
        for ep in self.episodes:
            vl = int(valid_len[ep])
            t_lo = (self.K - 1) * self.stride
            if self.padding_mode == "valid_only":
                t_hi = vl - 1 - (self.H - 1) * self.stride
            elif self.padding_mode == "half_dp":
                t_hi = vl - 1 - (self.H // 2) * self.stride
            else:  # "dp"
                t_hi = vl - 1
            if t_hi < t_lo:
                continue
            for t in range(t_lo, t_hi + 1):
                idx_list.append((int(ep_offset[ep]), t, vl))
        if not idx_list and split == "train":
            raise RuntimeError(
                f"No valid windows for split={split} with H={self.H}, K={self.K}, "
                f"stride={self.stride}, padding_mode={self.padding_mode}"
            )
        self.indices = np.asarray(idx_list, dtype=np.int64)

        # ---- fit normalizer from TRAIN frames (both splits use train stats) ----
        self.normalizer = self._build_normalizer()

        # ---- lazy per-worker h5 handle ----
        self._h5: Optional[h5py.File] = None
        self._latent = None

    # ------------------------------------------------------------------
    def _build_normalizer(self):
        """Pluggable: pick normalizer by kind, fit on TRAIN split frames."""
        with h5py.File(self.h5_path, "r") as f:
            ep_success = _read_success(f)
            ep_offset = f["ep_offset"][:]
            valid_len = _read_valid_len(f)
            success_eps = np.where(ep_success)[0]
            if self.max_episodes and self.max_episodes > 0:
                success_eps = success_eps[: int(self.max_episodes)]
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(len(success_eps))
            shuffled = success_eps[perm]
            n_val = 0 if self.val_frac <= 0 else max(1, int(round(len(shuffled) * self.val_frac)))  # val_frac<=0 -> no val (tiny-N)
            train_eps = np.sort(shuffled[n_val:])
            lat = f["latent"]
            frames = []
            for ep in train_eps:
                off = int(ep_offset[ep])
                vl = int(valid_len[ep])
                if vl <= 0:
                    continue
                frames.append(np.asarray(lat[off: off + vl], dtype=np.float32))
            arr = np.concatenate(frames, axis=0)
        print(f"[cube_latent] fitting {self.normalizer_kind!r} normalizer on "
              f"{arr.shape[0]} train frames × {arr.shape[1]} dims")
        return make_normalizer(self.normalizer_kind, arr)

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._latent = self._h5["latent"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        self._ensure_open()
        offset, t, vl = (int(v) for v in self.indices[idx])
        s = self.stride

        # history (K frames at stride s) — always in-bounds by t_lo constraint.
        hist = np.asarray(
            self._latent[offset + t - (self.K - 1) * s: offset + t + 1: s],
            dtype=np.float32,
        )                                                              # (K, D)

        # target (H frames at stride s) — valid_only path = no padding.
        if self.padding_mode == "valid_only":
            tgt = np.asarray(
                self._latent[offset + t: offset + t + self.H * s: s],
                dtype=np.float32,
            )                                                          # (H, D)
            is_pad = np.zeros(self.H, dtype=bool)
        else:  # dp / half_dp
            n_valid = max(0, min(self.H, (vl - 1 - t) // s + 1))
            tgt_parts = []
            if n_valid > 0:
                tgt_parts.append(np.asarray(
                    self._latent[offset + t: offset + t + n_valid * s: s],
                    dtype=np.float32,
                ))
            if n_valid < self.H:
                last_lat = np.asarray(
                    self._latent[offset + vl - 1: offset + vl], dtype=np.float32,
                )
                tgt_parts.append(np.repeat(last_lat, self.H - n_valid, axis=0))
            tgt = np.concatenate(tgt_parts, axis=0)
            is_pad = np.zeros(self.H, dtype=bool)
            if n_valid < self.H:
                is_pad[n_valid:] = True

        # Apply normalizer (per-dim) using the SAME stats for hist + target.
        hist = self.normalizer.normalize(hist).astype(np.float32)
        tgt = self.normalizer.normalize(tgt).astype(np.float32)

        cond = hist.reshape(-1).astype(np.float32)                     # (K*D,)
        return {
            "obs": torch.from_numpy(tgt),
            "cond": torch.from_numpy(cond),
            "is_pad": torch.from_numpy(is_pad),
        }
