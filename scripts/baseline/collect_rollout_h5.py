"""Live rollout -> play-format h5 recorder for the B-set (SuP fixed-k bootstrap).

Used by eval_unified.py when ++eval.collect_h5=true. Records each env's rollout
as one episode in the SAME layout the play corpora use, so the N-source corpus
merger can mix B with O/D without schema surgery.

Design (why not World.record_dataset): the eval loop owns stepping, injects
dataset init states, forbids resets, and runs history_size=3 — record_dataset
rejects all of that. So we stream (step, env)-indexed temp datasets during the
loop and rewrite episode-contiguous h5 at finalize.

Alignment: the eval loop captures AFTER world.step(), i.e. (o_{t+1}, a_t).
We pair the PREVIOUS capture's obs with the current action -> rows are exactly
(o_t, a_t). The first loop iteration only primes the obs buffer (row skipped:
pre-step reset obs is not fully populated for cube's monkey-patched
wrist_pixels), so each episode loses frame 0 — negligible vs ~300-630 budgets
and uniform across tasks. No NaN action rows are written (unlike swm's
left-shift convention).

Episode cut: an episode ends HOLD_TAIL rows after its first terminated=True
step (default 50) — the tail makes hold-N (settle) scoring possible downstream
and captures the post-latch settle dynamics (slippery latch != hold). The +1
inside the cut also keeps the latch STATE itself: the terminated flag stored
at row t describes the post-step state, which lands in row t+1's obs.
Never-terminated envs run the full budget. All episodes are KEPT (keep-all);
per-frame `terminated` (instantaneous in-tolerance, one-row lag as above) plus
per-episode `ep_success` columns let downstream slice/score.
"""

from pathlib import Path

import numpy as np
import hdf5plugin  # noqa: F401  (registers Blosc)
import h5py

_BLOSC = dict(hdf5plugin.Blosc(cname="lz4", clevel=5,
                               shuffle=hdf5plugin.Blosc.SHUFFLE))

# Per-task capture spec: world.infos key -> (h5 key, dtype).
# Vector keys arrive as (n_envs, history, d) -> we take [:, -1]; pixel keys as
# (n_envs, history, H, W, 3) or (n_envs, H, W, 3) (cube wrist monkey-patch).
TASK_KEYS = {
    "pusht": {
        "pixels": ("pixels", np.uint8),
        "proprio": ("proprio", np.float32),
        "state": ("state", np.float32),
    },
    "cube": {
        "pixels": ("pixels", np.uint8),
        "wrist_pixels": ("wrist_pixels", np.uint8),
        "observation": ("observation", np.float32),
        "proprio/gripper_opening": ("proprio_gripper_opening", np.float32),
        "proprio/effector_pos": ("proprio_effector_pos", np.float32),
        "proprio/gripper_contact": ("proprio_gripper_contact", np.float32),
        "privileged/block_0_pos": ("privileged_block_0_pos", np.float32),
    },
}


def _latest(arr):
    """Strip the history axis: (n, hist, ...) -> (n, ...); pass-through else."""
    a = np.asarray(arr)
    if a.ndim in (3, 5):  # (n,hist,d) / (n,hist,H,W,3)
        return a[:, -1]
    return a


class RolloutH5Recorder:
    def __init__(self, out_path, task_name, n_envs, budget, hold_tail=50):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp_path = self.out_path.with_suffix(".collecting.h5")
        if task_name not in TASK_KEYS:
            raise ValueError(f"collect_h5: unsupported task '{task_name}'")
        self.task = task_name
        self.keys = TASK_KEYS[task_name]
        self.n = int(n_envs)
        self.budget = int(budget)
        self.hold_tail = int(hold_tail)
        self._t = 0                      # rows written so far
        self._pending_obs = None         # obs snapshot from previous capture
        self._first_term = np.full(self.n, -1, dtype=np.int64)
        self._f = h5py.File(self.tmp_path, "w")
        self._ds = {}

    def _ensure(self, key, sample_shape, dtype):
        if key in self._ds:
            return
        shape = (self.budget, self.n, *sample_shape)
        if dtype == np.uint8 and len(sample_shape) == 3:
            chunks = (1, 1, *sample_shape)          # one (step, env) frame
            self._ds[key] = self._f.create_dataset(
                key, shape=shape, dtype=dtype, chunks=chunks, **_BLOSC)
        else:
            self._ds[key] = self._f.create_dataset(
                key, shape=shape, dtype=dtype,
                chunks=(min(256, self.budget), self.n, *sample_shape))

    def _snapshot(self, infos):
        obs = {}
        for src_key, (out_key, dtype) in self.keys.items():
            if src_key not in infos:
                raise KeyError(
                    f"collect_h5: world.infos missing '{src_key}' "
                    f"(available: {sorted(infos.keys())[:20]}...)")
            obs[out_key] = _latest(infos[src_key]).astype(dtype, copy=True)
        return obs

    def step(self, infos, action, terminateds):
        """Call once per loop iteration, AFTER world.step().

        infos: post-step world.infos (= o_{t+1}); action: a_t executed this
        step; terminateds: per-env terminated flags after this step.
        """
        cur_obs = self._snapshot(infos)
        if self._pending_obs is not None and self._t < self.budget:
            t = self._t
            if action is None:
                raise RuntimeError("collect_h5: executed action unavailable "
                                   "(policy wrapper did not stash it)")
            act = np.asarray(action, np.float32).reshape(self.n, -1)
            self._ensure("action", (act.shape[1],), np.float32)
            self._ds["action"][t] = act
            for out_key, arr in self._pending_obs.items():
                self._ensure(out_key, arr.shape[1:], arr.dtype)
                self._ds[out_key][t] = arr
            term = (np.zeros(self.n, bool) if terminateds is None
                    else np.asarray(terminateds, bool))
            self._ensure("terminated", (), np.bool_)
            self._ds["terminated"][t] = term
            newly = (self._first_term < 0) & term
            self._first_term[newly] = t
            self._t += 1
        self._pending_obs = cur_obs

    def finalize(self, attrs=None):
        """Rewrite (step, env) temp layout into episode-contiguous play format."""
        T = self._t
        self._f.flush()
        # Episode length per env: first-latch row + 1 (the latch STATE lands in
        # the next row, see module docstring) + hold_tail rows for settle/hold-N
        # scoring; capped at what was recorded.
        ep_len = np.where(
            self._first_term >= 0,
            np.minimum(self._first_term + 2 + self.hold_tail, T), T
        ).astype(np.int64)
        ep_off = np.zeros(self.n, np.int64)
        ep_off[1:] = np.cumsum(ep_len[:-1])
        total = int(ep_len.sum())

        with h5py.File(self.out_path, "w") as out:
            out.create_dataset("ep_len", data=ep_len.astype(np.int32))
            out.create_dataset("ep_offset", data=ep_off)
            out.create_dataset(
                "episode_idx", data=np.repeat(np.arange(self.n), ep_len))
            out.create_dataset(
                "step_idx",
                data=np.concatenate([np.arange(l) for l in ep_len]))
            out.create_dataset(
                "ep_success", data=(self._first_term >= 0))
            for key, src in self._ds.items():
                sample_shape = src.shape[2:]
                dtype = src.dtype
                if dtype == np.uint8 and len(sample_shape) == 3:
                    dst = out.create_dataset(
                        key, shape=(total, *sample_shape), dtype=dtype,
                        chunks=(1, *sample_shape), **_BLOSC)
                else:
                    dst = out.create_dataset(
                        key, shape=(total, *sample_shape), dtype=dtype)
                for e in range(self.n):
                    L = int(ep_len[e])
                    dst[ep_off[e]:ep_off[e] + L] = src[:L, e]
            for k, v in (attrs or {}).items():
                if v is not None:
                    out.attrs[k] = v
        self._f.close()
        self.tmp_path.unlink(missing_ok=True)
        return {"episodes": self.n, "frames": total,
                "success_eps": int((self._first_term >= 0).sum()),
                "path": str(self.out_path)}
