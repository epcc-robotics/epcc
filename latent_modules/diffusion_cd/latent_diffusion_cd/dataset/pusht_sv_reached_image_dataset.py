"""PushT SV image-DP over SAIL reached-pose labels: action = [x, y, precision].

Consumes the h5 that ``scripts/sail/make_labels_pusht.py`` writes, where

    action[i] = [pos_agent[i+1].x, pos_agent[i+1].y, precision[i]]

i.e. the reached target is ALREADY shifted by one — a policy acting at frame i
should command ``action[i]``.

Why this class exists at all. ``PushTSVImageDataset`` would load this file fine
(it only needs ep_offset / ep_len / action / pixels, and the label h5 supplies
pixels as an ExternalLink), but its ``__getitem__`` takes the action window
straight from the sampler:

    action = sample["action"]          # -> label rows t .. t+horizon-1

The observation window is ``t .. t+n_obs_steps-1``, so the frame the policy acts
from is ``a = t + n_obs_steps - 1``, not ``t``. Feeding it row ``t`` hands it the
target for a frame it has already passed — the same off-by-one that cost the
SkiP port a debugging round (docs #6) and that
``pusht_sv_abs_image_dataset`` documents in its own header. Here the fix is a
shift, not a +1: the label file already did the +1, so the correct slice starts
at ``a``.

Everything else — normalizer, image pipeline, episode handling — is inherited.
The MinMax action normalizer the parent fits covers all three columns, which is
what we want: x/y in pixels and precision in {0,1} are each mapped to [-1,1].
"""
from typing import Dict

import h5py
import numpy as np
import torch
from cleandiffuser.dataset.dataset_utils import dict_apply

from latent_diffusion_cd.dataset.pusht_sv_image_dataset import PushTSVImageDataset


class PushTSVReachedImageDataset(PushTSVImageDataset):
    def __init__(self, dataset_dir, shape_meta, n_obs_steps=2, horizon=16,
                 pad_before=0, pad_after=0, max_episodes=None):
        super().__init__(dataset_dir, shape_meta, n_obs_steps, horizon,
                         pad_before, pad_after, max_episodes)
        self._build_action_meta(dataset_dir, max_episodes)

    def _build_action_meta(self, path, max_ep):
        """Concatenate per-episode `action` in the SAME episode order the replay
        buffer used, so global indices from `self.sampler` address it."""
        with h5py.File(path, "r") as f:
            ep_off = f["ep_offset"][:]
            ep_len = f["ep_len"][:]
            if max_ep is not None:
                ep_off = ep_off[:int(max_ep)]
                ep_len = ep_len[:int(max_ep)]
            act = f["action"][:].astype(np.float32)        # (N, 3) [x, y, precision]
        parts, ends, g0 = [], [], 0
        for i in range(len(ep_len)):
            lo, ln = int(ep_off[i]), int(ep_len[i])
            parts.append(act[lo:lo + ln])
            g0 += ln
            ends.append(g0)
        self._reached = np.nan_to_num(np.concatenate(parts), nan=0.0).astype(np.float32)
        self._ep_ends = np.asarray(ends, dtype=np.int64)
        lo_, hi_ = self._reached.min(0), self._reached.max(0)
        print(f"[PushTSVReachedImageDataset] T={len(self._reached)}  action=reached(2)+precision(1)  "
              f"x[{lo_[0]:.1f},{hi_[0]:.1f}] y[{lo_[1]:.1f},{hi_[1]:.1f}] "
              f"precision_frac={self._reached[:, 2].mean():.3f}")

    def _ep_end_for(self, t):
        return int(self._ep_ends[np.searchsorted(self._ep_ends, t, side="right")])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        obs_dict = {}
        T_obs = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
        for k in self.rgb_keys:
            img = sample[k][:T_obs].astype(np.float32) / 255.0
            img = self.normalizer["obs"][k].normalize(img)
            img = np.moveaxis(img, -1, -3).astype(np.float32)
            obs_dict[k] = img
        for k in self.lowdim_keys:
            v = sample[k][:T_obs].astype(np.float32)
            v = self.normalizer["obs"][k].normalize(v).astype(np.float32)
            obs_dict[k] = v

        t = int(self.sampler.indices[idx][0])
        a = t + self.n_obs_steps - 1          # frame the policy acts from
        ep_end = self._ep_end_for(a)
        ii = np.clip(np.arange(a, a + self.horizon), 0, ep_end - 1)
        action = self._reached[ii].astype(np.float32)     # (horizon, 3)
        action = self.normalizer["action"].normalize(action).astype(np.float32)
        return dict_apply({"obs": obs_dict, "action": action}, torch.tensor)
