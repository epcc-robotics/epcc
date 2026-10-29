"""PushT SV image-DP whose ACTION is the ABSOLUTE agent pose, with no relabel.

Why this exists (2026-07-31). Slippery PushT was moved off the
Robomimic DP repo onto CleanDiffuser while keeping
its absolute action abstraction, so that DP / SuP / SkiP / SAIL inside Slippery
all speak the same abstraction. CleanDiffuser had a delta variant
(`pusht_sv_image_dataset`) and an absolute-WITH-SkiP-relabel variant
(`pusht_sv_skip_image_dataset`); the plain absolute one was missing.

    PushTSVImageDataset          action = 2-d push DELTA          (f["action"])
    PushTSVSkipImageDataset      action = ABSOLUTE pose + SkiP relabel (leaps)
    PushTSVAbsImageDataset       action = ABSOLUTE pose, no relabel  <- here

Nothing is recomputed: `pos_agent` is already stored in the demo h5. This class
only (a) points the action at that column and (b) refits the action normalizer
to it, exactly as the SkiP dataset does — minus the MSK segmentation.

🔑 The horizon target is `a + 1`, not `t`. Frames `t .. t+n_obs-1` are the
observation; the current frame is `a = t + n_obs - 1`. At inference the executed
action is `pred[0]` while the env sits at `pose[a]`, so the first target must be
the NEXT pose. Using `t` would make the first action point one step behind the
current pose — a backward/no-op step. This is the off-by-one that cost the SkiP
port a debugging round (docs #6); it is inherited here deliberately.
"""
from typing import Dict

import h5py
import numpy as np
import torch
from cleandiffuser.dataset.dataset_utils import MinMaxNormalizer, dict_apply

from latent_diffusion_cd.dataset.pusht_sv_image_dataset import PushTSVImageDataset


class PushTSVAbsImageDataset(PushTSVImageDataset):
    def __init__(self, dataset_dir, shape_meta, n_obs_steps=2, horizon=16,
                 pad_before=0, pad_after=0, max_episodes=None):
        super().__init__(dataset_dir, shape_meta, n_obs_steps, horizon,
                         pad_before, pad_after, max_episodes)
        self._build_abs_meta(dataset_dir, max_episodes)
        # Refit the action normalizer: the parent fitted MinMax on 2-d deltas
        # (range ~[-1,1]); absolute agent poses live in [0,512] px.
        self.normalizer["action"] = MinMaxNormalizer(self._abs_pose)

    def _build_abs_meta(self, path, max_ep):
        """Concatenate per-episode `pos_agent` in the SAME episode order the
        replay buffer used, so global indices from `self.sampler` address it."""
        with h5py.File(path, "r") as f:
            ep_off = f["ep_offset"][:]
            ep_len = f["ep_len"][:]
            if max_ep is not None:
                ep_off = ep_off[:int(max_ep)]
                ep_len = ep_len[:int(max_ep)]
            agent = f["pos_agent"][:].astype(np.float32)      # (N, 2) pusher xy
        abs_l, ends, g0 = [], [], 0
        for i in range(len(ep_len)):
            lo, ln = int(ep_off[i]), int(ep_len[i])
            abs_l.append(agent[lo:lo + ln])
            g0 += ln
            ends.append(g0)
        # H5 padding zones can carry NaN; the parent does the same for actions.
        self._abs_pose = np.nan_to_num(np.concatenate(abs_l), nan=0.0).astype(np.float32)
        self._ep_ends = np.asarray(ends, dtype=np.int64)
        lo_, hi_ = self._abs_pose.min(0), self._abs_pose.max(0)
        print(f"[PushTSVAbsImageDataset] T={len(self._abs_pose)}  action=abs-agent-pos(2)  "
              f"range x[{lo_[0]:.1f},{hi_[0]:.1f}] y[{lo_[1]:.1f},{hi_[1]:.1f}]")

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
        a = t + self.n_obs_steps - 1          # current frame the policy acts from
        ep_end = self._ep_end_for(a)
        ii = np.clip(np.arange(a + 1, a + 1 + self.horizon), 0, ep_end - 1)
        action = self._abs_pose[ii].astype(np.float32)        # (horizon, 2) abs agent pos
        action = self.normalizer["action"].normalize(action).astype(np.float32)
        return dict_apply({"obs": obs_dict, "action": action}, torch.tensor)
