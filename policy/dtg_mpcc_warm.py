"""DTG-MPCC with the first-replan history ring seeded from real frames.

Port of the warm-ring fix from the upstream latent-MPCC code (2026-09-07), verified
against our own copy of the defect.

# The defect

`eval_diffusion_mpcc.DiffusionMPCCPolicy._update_ring` (our copy, :145-152) initialises the
history ring on the FIRST call by repeating the current encoding into all `ring_len` slots:

    if self._ring is None:
        self._ring = cur_emb.unsqueeze(1).repeat(1, self._ring_len, 1)

`_refs_from_ring` then reads slots `_stride_indices` (:178, :267), so at replan 0 the
generator is conditioned on a history whose frames are all identical — a zero-motion
sequence that never occurs in training. From replan 1 (one `receding_horizon * action_block`
= 15 env-steps later) the ring holds real encodings, so the damage is confined to the first
plan; but MPCC tracks that first plan through the approach, which for cube is exactly the
part that decides the grasp.

Sizes differ between generator families and both are affected:
    K=2, stride=5  -> ring_len 6, indices [0, 5]  -> history [z_t, z_t]
    K=6, stride=1  -> ring_len 6, indices [0..5]  -> history [z_t] * 6   (our h84 models)

We measured the cost independently before seeing the upstream handoff: generating from a
duplicated history instead of the real one multiplies the per-block latent error by
1.24-2.14 across all three of our world-model arms (256 held-out windows, cube):
    AW   agent 2.14  wrist 1.24
    AWP  agent 1.95  wrist 1.35  proprio 1.94
    AP   agent 2.01               proprio 2.13
Upstream reports cube DTG-MPCC SR .70 -> .90 from fixing it, with the generator, world
model, solver and protocol untouched.

# The fix

Seed the ring, before the first `_update_ring`, with the `ring_len - 1` encodings of the
env-steps PRECEDING the reset frame, so that after the policy's own first roll-and-append the
ring reads [z_{s-(ring_len-1)}, ..., z_s] and `_stride_indices` land exactly as in training.

In sim the reset frame is dataset frame `eval.start_offset` of a demo episode, so those
frames exist and their encodings are already cached — the same latent file the generator was
trained on, produced by the same (frozen) encoder the eval loads. That makes this a pure
control: nothing changes except the first plan's conditioning.

🔴 `warm_ring.latent_h5` MUST be the cache built with THIS cell's world model. The cube arms
are 384-d (AW, AP) and 576-d (AWP); a width mismatch is caught below, but AW and AP are the
same width and would not be, so the config carries the path explicitly rather than guessing.

For deployment (no dataset) the equivalent is to spend the first `ring_len - 1` env-steps
filling the ring from live observations before the first replan, or to train the generator
with duplicated-history augmentation so the cold start is in-distribution.
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import torch
from omegaconf import DictConfig

from policy.dtg_mpcc import DTGMPCCPolicy


def seed_warm_ring(policy: DTGMPCCPolicy, cfg: DictConfig | None, eval_world: Any) -> bool:
    """Seed ``policy._inner._ring`` from the eval episodes' cached latents.

    Shared by :class:`DTGMPCCWarmPolicy` and by
    ``policy.dtg_mpcc_cand_warm.DTGMPCCCandidateWarmPolicy`` so the two families apply
    byte-identical seeding. Must run after the policy's own ``reset`` (which builds
    ``_inner``) and before its first ``get_action`` (whose ``_update_ring`` would otherwise
    duplicate the current frame). Returns True when the ring was seeded, False when
    ``warm_ring`` is absent/disabled — the caller then runs the duplicated cold start, and
    the only trace of that is the printed line, so grep for ``[warm-ring] seeded``.
    """
    wr = (cfg or {}).get("warm_ring", None)
    if wr is None:
        # A *Warm policy was explicitly requested; a missing block would otherwise run the
        # exact cold start it exists to fix and finish rc=0 under a warm run name.
        raise ValueError(
            "warm policy selected but cfg.warm_ring is absent — add the block to the cell "
            "yaml (see cube_dtg_mpcc_warm.yaml) or pass +warm_ring.enabled=false to run cold "
            "deliberately")
    if not wr.get("enabled", True):
        print("[warm-ring] disabled (warm_ring.enabled=false) — duplicated cold start")
        return False
    h5_path = str(wr["latent_h5"])

    inner = policy._inner
    ring_len = int(inner._ring_len)
    n_seed = ring_len - 1
    eps = np.asarray(eval_world.eval_episodes, dtype=np.int64)
    starts = np.asarray(eval_world.eval_start_idx, dtype=np.int64)

    with h5py.File(h5_path, "r") as f:
        lat = f["latent"]
        D_h5 = int(lat.shape[1])
        offs = np.asarray(f["ep_offset"][:], dtype=np.int64)
        seed = []
        for e, s in zip(eps, starts):
            o = int(offs[int(e)])
            lo = max(o + int(s) - n_seed, o)
            frames = np.asarray(lat[lo:o + int(s)], dtype=np.float32)
            if len(frames) == 0:                    # reset frame is the episode start
                frames = np.asarray(lat[o:o + 1], dtype=np.float32)
            if len(frames) < n_seed:                # near the episode start: repeat oldest
                pad = np.repeat(frames[:1], n_seed - len(frames), axis=0)
                frames = np.concatenate([pad, frames], axis=0)
            seed.append(frames)
    seed = torch.as_tensor(np.stack(seed))          # (n_env, ring_len-1, D)

    # 🔴 Width check. It catches an AWP cache handed to an AW/AP cell, but NOT an AP
    # cache handed to an AW cell — both are 384-d. That pairing is the config's job.
    D_model = int(inner._diff.obs_dim)
    if D_h5 != D_model:
        raise RuntimeError(
            f"warm_ring.latent_h5 has D={D_h5} but the generator has obs_dim={D_model}. "
            f"The seed cache must come from this cell's world model: {h5_path}"
        )

    dev = getattr(inner._diff, "device", "cuda")
    # The policy's own first _update_ring rolls left (dropping slot 0) and writes the live
    # encoding into slot -1, so prepend a throwaway copy of the OLDEST frame: the seed then
    # lands in slots 1..ring_len-1 and slot 0 is what gets dropped.
    ring = torch.cat([seed[:, :1], seed], dim=1).to(dev).contiguous()
    assert ring.shape[1] == ring_len, f"ring {tuple(ring.shape)} != ring_len {ring_len}"
    inner._ring = ring

    d01 = (seed[:, 0] - seed[:, -1]).abs().amax(dim=-1)
    print(f"[warm-ring] seeded {ring.shape} from {h5_path}")
    print(f"[warm-ring]   ring_len={ring_len} n_seed={n_seed} "
          f"stride_indices={list(inner._stride_indices)}")
    print(f"[warm-ring]   |oldest - newest| over envs: median {d01.median():.4f} "
          f"min {d01.min():.4f} max {d01.max():.4f}   (cold start would be 0.0000)")
    return True


class DTGMPCCWarmPolicy(DTGMPCCPolicy):
    """DTGMPCCPolicy whose first plan is conditioned on real history."""

    def reset(self, world: Any, cfg: DictConfig | None = None,
              eval_world: Any | None = None) -> None:
        super().reset(world, cfg, eval_world)          # full DTG setup, untouched
        seed_warm_ring(self, cfg, eval_world)


