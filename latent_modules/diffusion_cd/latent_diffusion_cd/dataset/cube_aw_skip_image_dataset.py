"""SkiP-relabeled cube AW image dataset (SkiP baseline port).

Subclass of CubeAWImageDataset. Two changes vs the base:
  1. ACTION = absolute ee pose [eff_xyz(3), eff_yaw(1), gripper(1)] instead of
     the 5-d delta — needed for SkiP's "leap to a distant pose" to be a
     position-independent, controller-reachable target.
  2. TARGET RELABEL (SkiP/MSK): for a skip-region timestep the target chunk
     starts at the next key-segment entrance (a leap); for a key-region
     timestep it starts at the current step (dense). obs is UNCHANGED.

MSK segmentation (ported from ogbench cube_markov / SkiP) runs on the effector
POSITION path (our actions are deltas, so the ee path carries the motion).
Additive: base CubeAWImageDataset + delta pipeline untouched.
"""
from typing import Dict

import h5py
import numpy as np
import torch
from scipy.fft import dct

from cleandiffuser.dataset.dataset_utils import MinMaxNormalizer, dict_apply
from latent_diffusion_cd.dataset.cube_aw_image_dataset import CubeAWImageDataset

# ---- MSK constants (SkiP defaults) ----
_W, _HFF, _Q, _MIN, _BEND_HALF, _KF_R = 16, 0.5, 0.75, 3, 2, 5
# velocity-stop keyframe threshold: fraction of the episode median step speed
# below which the manipulator is considered "stopped" (SkiP _is_stopped analog).
# The original absolute joint-vel tol (0.01) is not transferable to our
# position-space signal, so we recalibrate scale-relative. Tuned via sweep.
_STOP_FRAC = 0.25


def _freq_metrics(traj, window=_W, hff=_HFF):
    T = traj.shape[0]
    vel = np.diff(traj, axis=0, prepend=traj[0:1])
    hr = np.zeros(T); bend = np.zeros(T); half = window // 2
    for t in range(T):
        s = max(0, t - half); e = min(T, s + window); s = max(0, e - window)
        sv = vel[s:e]; sp = traj[s:e]; L = sv.shape[0]
        if L <= 1:
            continue
        E = np.sum(dct(sv, axis=0, norm="ortho") ** 2, axis=1); tot = float(E.sum())
        if tot > 1e-12:
            c = int(np.clip(round((1 - hff) * L), 0, L - 1)); hr[t] = float(E[c:].sum()) / tot
        if L >= 3:
            d = sp[-1] - sp[0]; nd = np.linalg.norm(d)
            if nd > 1e-8:
                u = d / nd; rel = sp - sp[0]; perp = np.linalg.norm(rel - (rel @ u)[:, None] * u, axis=1)
                bend[t] = float(perp.mean() / (np.linalg.norm(np.diff(sp, axis=0), axis=1).mean() + 1e-8))
    return hr, bend


def _segs(mask, mn):
    out = []; T = len(mask); ins = False; st = 0
    for t in range(T):
        if mask[t] and not ins:
            ins = True; st = t
        elif not mask[t] and ins:
            ins = False
            if t - st >= mn:
                out.append((st, t - 1))
    if ins and T - st >= mn:
        out.append((st, T - 1))
    return out


def _stop_keyframes(pos, stop_frac=_STOP_FRAC, buffer=4):
    """Velocity-stop keyframes = frames where the manipulator dwells (per-step
    speed ~ 0), the position-space analog of SkiP's `_is_stopped`
    (joint_velocities ~ 0; rlbench_env.py:215). The original absolute joint-vel
    tolerance (0.01) does not transfer to our ee/agent-position signal, so we
    threshold step speed at `stop_frac` of the episode median step speed.
    Mirrors keypoint_discovery's debounce (stopped_buffer=4) + end guards
    (i != 0, i != T-2), and also emits the last frame as a KF
    (rlbench_env.py:254-256). Returns sorted keyframe indices.

    (We omit the original's gripper-stable guard: gripper transitions are added
    to the region separately, so a stop coinciding with one only re-covers the
    same frames.)
    """
    T = pos.shape[0]
    if T < 3:
        return [T - 1] if T > 0 else []
    speed = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[0:1]), axis=1)
    nz = speed[speed > 1e-9]
    thr = stop_frac * (float(np.median(nz)) if nz.size else 1.0)
    kfs = []
    sbuf = 0
    for i in range(T):
        stopped = (sbuf <= 0) and (speed[i] < thr) and (i != 0) and (i != T - 2)
        sbuf = buffer if stopped else sbuf - 1
        if stopped:
            kfs.append(i)
    kfs.append(T - 1)   # last-frame keyframe (rlbench_env.py:254)
    return kfs


def compute_region_next_high(eff_xyz, gripper, window=_W, q=_Q, stop_frac=_STOP_FRAC):
    T = eff_xyz.shape[0]
    hr, bend = _freq_metrics(eff_xyz, window=window)
    region = np.zeros(T, dtype=np.int64)
    if np.any(hr):
        for s, e in _segs(hr >= float(np.quantile(hr, q)), _MIN):
            region[s:e + 1] = 1
    vb = bend[bend > 0]
    if vb.size:
        bt = float(np.quantile(vb, q))
        for s, e in _segs(bend >= bt, 1):
            c = (s + e) // 2; region[max(0, c - _BEND_HALF):min(T, c + _BEND_HALF + 1)] = 1
    g = np.asarray(gripper, float)
    if g.max() - g.min() > 0.05:
        mid = 0.5 * (g.min() + g.max())
        for c in np.where(np.diff((g > mid).astype(int)) != 0)[0]:
            region[max(0, c - _KF_R):min(T, c + _KF_R + 1)] = 1
    # velocity-stop keyframes (SkiP _is_stopped analog): dwell points with no
    # gripper event — protects contact/settle frames. ±_KF_R, like the heuristic
    # KF dilation in prepare_skip_metadata (rlbench_dataset.py:160-169).
    for c in _stop_keyframes(eff_xyz, stop_frac=stop_frac):
        region[max(0, c - _KF_R):min(T, c + _KF_R + 1)] = 1
    seg_starts = np.array(sorted(set(s for s, _ in _segs(region == 1, 1))))
    next_high = np.full(T, -1, dtype=np.int64)
    for t in range(T):
        if region[t] == 0:
            fut = seg_starts[seg_starts > t]
            next_high[t] = int(fut[0]) if fut.size else -1
    return region, next_high


class CubeAWSkipImageDataset(CubeAWImageDataset):
    def __init__(self, dataset_dir, shape_meta, n_obs_steps=2, horizon=16,
                 pad_before=0, pad_after=0, max_episodes=None,
                 skip_window=_W, skip_quantile=_Q, stop_frac=_STOP_FRAC,
                 balanced_sampling=True):
        super().__init__(dataset_dir, shape_meta, n_obs_steps, horizon,
                         pad_before, pad_after, max_episodes)
        self._balanced = bool(balanced_sampling)
        self._build_skip_meta(dataset_dir, max_episodes, skip_window,
                              skip_quantile, stop_frac)
        # re-fit action normalizer to ABSOLUTE pose (was delta)
        self.normalizer["action"] = MinMaxNormalizer(self._abs_pose)

    def _build_skip_meta(self, path, max_ep, W, q, stop_frac=_STOP_FRAC):
        with h5py.File(path, "r") as f:
            ep_off = f["ep_offset"][:]; ep_len = f["ep_len"][:]
            if max_ep is not None:
                ep_off = ep_off[:int(max_ep)]; ep_len = ep_len[:int(max_ep)]
            eff = f["proprio_effector_pos"][:]; yaw = f["proprio_effector_yaw"][:]
            grip = f["proprio_gripper_opening"][:]
        abs_l, reg_l, nh_l, ends = [], [], [], []; g0 = 0
        for i in range(len(ep_len)):
            lo, ln = int(ep_off[i]), int(ep_len[i])
            e = eff[lo:lo + ln]; y = yaw[lo:lo + ln]; gp = grip[lo:lo + ln]
            abs_l.append(np.concatenate([e, y, gp], axis=1))
            region, nh = compute_region_next_high(e, gp[:, 0], W, q, stop_frac)
            reg_l.append(region)
            nh_l.append(np.where(nh >= 0, nh + g0, -1))
            g0 += ln; ends.append(g0)
        self._abs_pose = np.nan_to_num(np.concatenate(abs_l), nan=0.0).astype(np.float32)
        self._region = np.concatenate(reg_l)
        self._next_high = np.concatenate(nh_l)
        self._ep_ends = np.asarray(ends, dtype=np.int64)
        self._build_balanced_pools()
        print(f"[CubeAWSkipImageDataset] T={len(self._region)} key_frac="
              f"{self._region.mean():.0%} action=abs-pose(5)  W={W} q={q} "
              f"stop_frac={stop_frac} balanced={self._balanced} "
              f"(key_js={len(self._key_js)} skip_js={len(self._skip_js)})")

    def _build_balanced_pools(self):
        """SkiP samples 50/50 from skip vs key anchors (rlbench_dataset.py:558-566).
        SequenceSampler gives uniform windows, so we bucket sampler indices by the
        region of their anchor a = start + n_obs_steps - 1 and draw 50/50 in
        __getitem__. A window is a valid SKIP sample only if next_high[a] >= 0.
        """
        n = len(self.sampler.indices)
        anchors = np.clip(
            np.array([int(self.sampler.indices[j][0]) for j in range(n)])
            + self.n_obs_steps - 1, 0, len(self._region) - 1)
        reg = self._region[anchors]; nh = self._next_high[anchors]
        self._key_js = np.where(reg == 1)[0]
        self._skip_js = np.where((reg == 0) & (nh >= 0))[0]

    def _ep_end_for(self, t):
        return int(self._ep_ends[np.searchsorted(self._ep_ends, t, side="right")])

    def get_normalizer(self):
        # base builds obs + a (delta) action normalizer; action normalizer is
        # overwritten to abs-pose in __init__ after _build_skip_meta.
        return super().get_normalizer()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # SkiP 50/50 skip-vs-key sampling (rlbench_dataset.py:558-566): override
        # the uniform window idx with a balanced draw. Marginal 50/50 holds even
        # if DataLoader workers share the np.random seed (as in the original).
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

        # --- SkiP relabel: leap target for skip steps, dense for key steps ---
        # `t` = window START (global step). obs = frames [t .. t+n_obs_steps-1],
        # so the CURRENT frame (latest obs) is the anchor a = t+n_obs_steps-1 —
        # this is what SkiP calls `idx`. At inference the executed action is
        # abs_pred[0] (offset 0) and the env sits at pose[a], so:
        #   key  -> target must be the NEXT pose after current: a+1
        #   skip -> leap to next-key entrance decided AT the current frame a
        # (Using `t` here, as before, put the first key-target at pose[t] = one
        #  step BEHIND current -> a backward/no-op first action. See docs #6.)
        t = int(self.sampler.indices[idx][0])          # window start (global step)
        a = t + self.n_obs_steps - 1                    # current frame = SkiP's idx
        if self._region[a] == 0 and self._next_high[a] >= 0:
            tgt = int(self._next_high[a])               # skip -> leap to next key
        else:
            tgt = a + 1                                 # key  -> next pose after current
        ep_end = self._ep_end_for(a)
        ii = np.clip(np.arange(tgt, tgt + self.horizon), 0, ep_end - 1)
        action = self._abs_pose[ii].astype(np.float32)  # (horizon, 5) abs pose
        action = self.normalizer["action"].normalize(action).astype(np.float32)
        return dict_apply({"obs": obs_dict, "action": action}, torch.tensor)
