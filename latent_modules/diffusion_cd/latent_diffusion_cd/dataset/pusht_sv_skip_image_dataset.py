"""SkiP-relabeled PushT SV image dataset (SkiP baseline port to PushT).

Subclass of PushTSVImageDataset. Two changes vs the base:
  1. ACTION = absolute AGENT position [x, y] (2-d) instead of the 2-d push
     delta — SkiP's "leap to a distant pose" needs a position-independent,
     controller-reachable target (mirrors cube's abs-ee-pose refit).
  2. TARGET RELABEL (SkiP/MSK): for a skip-region timestep the target chunk
     starts at the next key-segment entrance (a leap); for a key-region
     timestep it starts at the current step (dense). obs is UNCHANGED.

MSK segmentation runs on the AGENT POSITION path (`pos_agent`, the pusher xy).
PushT has NO gripper, so the gripper-transition keyframe anchor used for cube is
absent — but the VELOCITY-STOP keyframes (SkiP `_is_stopped` analog) ARE kept:
they fire without any gripper event, so they are the sole heuristic-KF
protection for this non-prehensile task (mirrors why the original SkiP handles
slide_block etc.). Region = DCT high-freq energy OR bend OR velocity-stop KF.
The MSK core (`_freq_metrics`, `_segs`, `_stop_keyframes`) is reused verbatim
from the cube port so the two baselines share identical spectral logic.

Additive: base PushTSVImageDataset + 2-d delta pipeline untouched.
"""
from typing import Dict

import h5py
import numpy as np
import torch

from cleandiffuser.dataset.dataset_utils import MinMaxNormalizer, dict_apply
from latent_diffusion_cd.dataset.pusht_sv_image_dataset import PushTSVImageDataset
# Reuse the exact MSK spectral core from the cube SkiP port (task-agnostic:
# operates on a position trajectory) so both baselines stay in lock-step.
from latent_diffusion_cd.dataset.cube_aw_skip_image_dataset import (
    _freq_metrics, _segs, _stop_keyframes, _W, _Q, _MIN, _BEND_HALF, _KF_R,
    _STOP_FRAC,
)


def compute_region_next_high_nogrip(agent_xy, window=_W, q=_Q, stop_frac=_STOP_FRAC):
    """Key/skip regions from the agent (pusher) xy path — NO gripper anchor.

    Same as cube ``compute_region_next_high`` minus the gripper-transition block
    (PushT has no gripper), but WITH the velocity-stop keyframes — the only
    heuristic-KF protection a non-prehensile task can have. Key = high DCT-freq
    energy OR high bend OR velocity-stop; everything else is skip.
    """
    T = agent_xy.shape[0]
    hr, bend = _freq_metrics(agent_xy, window=window)
    region = np.zeros(T, dtype=np.int64)
    if np.any(hr):
        for s, e in _segs(hr >= float(np.quantile(hr, q)), _MIN):
            region[s:e + 1] = 1
    vb = bend[bend > 0]
    if vb.size:
        bt = float(np.quantile(vb, q))
        for s, e in _segs(bend >= bt, 1):
            c = (s + e) // 2
            region[max(0, c - _BEND_HALF):min(T, c + _BEND_HALF + 1)] = 1
    # velocity-stop keyframes (SkiP _is_stopped analog) — sole heuristic KF for
    # non-prehensile pushT: protects contact/settle frames the pusher dwells at.
    for c in _stop_keyframes(agent_xy, stop_frac=stop_frac):
        region[max(0, c - _KF_R):min(T, c + _KF_R + 1)] = 1
    seg_starts = np.array(sorted(set(s for s, _ in _segs(region == 1, 1))))
    next_high = np.full(T, -1, dtype=np.int64)
    for t in range(T):
        if region[t] == 0:
            fut = seg_starts[seg_starts > t]
            next_high[t] = int(fut[0]) if fut.size else -1
    return region, next_high


class PushTSVSkipImageDataset(PushTSVImageDataset):
    def __init__(self, dataset_dir, shape_meta, n_obs_steps=2, horizon=16,
                 pad_before=0, pad_after=0, max_episodes=None,
                 skip_window=_W, skip_quantile=_Q, stop_frac=_STOP_FRAC,
                 balanced_sampling=True):
        super().__init__(dataset_dir, shape_meta, n_obs_steps, horizon,
                         pad_before, pad_after, max_episodes)
        self._balanced = bool(balanced_sampling)
        self._build_skip_meta(dataset_dir, max_episodes, skip_window,
                              skip_quantile, stop_frac)
        # re-fit action normalizer to ABSOLUTE agent pos (was 2-d delta)
        self.normalizer["action"] = MinMaxNormalizer(self._abs_pose)

    def _build_skip_meta(self, path, max_ep, W, q, stop_frac=_STOP_FRAC):
        with h5py.File(path, "r") as f:
            ep_off = f["ep_offset"][:]; ep_len = f["ep_len"][:]
            if max_ep is not None:
                ep_off = ep_off[:int(max_ep)]; ep_len = ep_len[:int(max_ep)]
            agent = f["pos_agent"][:].astype(np.float32)      # (N, 2) pusher xy
        abs_l, reg_l, nh_l, ends = [], [], [], []; g0 = 0
        for i in range(len(ep_len)):
            lo, ln = int(ep_off[i]), int(ep_len[i])
            a = agent[lo:lo + ln]                             # (ln, 2)
            abs_l.append(a)
            region, nh = compute_region_next_high_nogrip(a, W, q, stop_frac)
            reg_l.append(region)
            nh_l.append(np.where(nh >= 0, nh + g0, -1))
            g0 += ln; ends.append(g0)
        self._abs_pose = np.nan_to_num(np.concatenate(abs_l), nan=0.0).astype(np.float32)
        self._region = np.concatenate(reg_l)
        self._next_high = np.concatenate(nh_l)
        self._ep_ends = np.asarray(ends, dtype=np.int64)
        self._build_balanced_pools()
        print(f"[PushTSVSkipImageDataset] T={len(self._region)} key_frac="
              f"{self._region.mean():.0%} action=abs-agent-pos(2)  W={W} q={q} "
              f"stop_frac={stop_frac} balanced={self._balanced} "
              f"(key_js={len(self._key_js)} skip_js={len(self._skip_js)})")

    def _build_balanced_pools(self):
        """50/50 skip-vs-key anchor sampling (rlbench_dataset.py:558-566)."""
        n = len(self.sampler.indices)
        anchors = np.clip(
            np.array([int(self.sampler.indices[j][0]) for j in range(n)])
            + self.n_obs_steps - 1, 0, len(self._region) - 1)
        reg = self._region[anchors]; nh = self._next_high[anchors]
        self._key_js = np.where(reg == 1)[0]
        self._skip_js = np.where((reg == 0) & (nh >= 0))[0]

    def _ep_end_for(self, t):
        return int(self._ep_ends[np.searchsorted(self._ep_ends, t, side="right")])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._balanced and len(self._key_js) and len(self._skip_js):
            pool = self._key_js if np.random.rand() < 0.5 else self._skip_js
            idx = int(pool[np.random.randint(len(pool))])
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

        # --- SkiP relabel (mirrors cube port, incl. the #6 anchor fix) ---
        # obs = frames [t .. t+n_obs-1]; current frame a = t+n_obs-1 = SkiP's idx.
        # At inference the executed action is abs_pred[0] (offset 0) and the env
        # sits at pose[a], so: key -> a+1 (next pose); skip -> next-key entrance
        # decided AT frame a. (tgt=t would put the first key-target one step
        # behind current = backward/no-op first action. See docs #6.)
        t = int(self.sampler.indices[idx][0])
        a = t + self.n_obs_steps - 1
        if self._region[a] == 0 and self._next_high[a] >= 0:
            tgt = int(self._next_high[a])
        else:
            tgt = a + 1
        ep_end = self._ep_end_for(a)
        ii = np.clip(np.arange(tgt, tgt + self.horizon), 0, ep_end - 1)
        action = self._abs_pose[ii].astype(np.float32)        # (horizon, 2) abs agent pos
        action = self.normalizer["action"].normalize(action).astype(np.float32)
        return dict_apply({"obs": obs_dict, "action": action}, torch.tensor)
