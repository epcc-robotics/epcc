"""Oracle MPCC cost model: real-env dynamics + the EXACT MPCC latent cost.

⚠️ UNFINISHED EXPERIMENT (WIP, 2026-06) — preliminary diagnostic, NOT a final
result. Only variant A is implemented; the confirmatory variant B (state-space /
block-pose cost instead of latent contour) is NOT yet run, so no conclusion is
committed here. Privileged + NON-DEPLOYABLE. Kept in-tree to continue the
investigation; default-OFF (enable via cost_weights.oracle_realenv.enable).

Variant A of the WM-as-bottleneck test. Subclasses ``MPCCCostModel`` and
overrides ONLY ``_predict_emb``: instead of rolling the JEPA predictor, it rolls
the *real* slippery-PushT physics for every CEM candidate and encodes the
resulting frames through the SAME JEPA encoder. Everything downstream — z_ref(θ),
ν, contour, progress — is inherited verbatim from ``MPCCCostModel.get_cost``.

If SR with this oracle ≫ SR with the WM predictor (at matched CEM budget), the
WM forward-dynamics error is the dominant cause of low MPCC SR.

NON-DEPLOYABLE diagnostic (the planner gets to try candidates in the true env).

State reconstruction: the live obs ``state`` = [agent_xy, block_xy, angle,
agent_vel_xy] has NO block velocity/angvel (needed for slippery coast). We
recover them by finite-differencing the state history (T_hist frames) and
self-calibrating Δt from the agent (whose pos AND vel are both observed).
"""
import numpy as np
import os
import multiprocessing as mp
import torch
from torchvision.transforms import v2 as TT
import stable_pretraining as spt

from latent_modules.mpcc.cost import MPCCCostModel
from latent_modules.mpcc import oracle_envpool as _ep
from perturb_replay_sensitivity import make_env

_TF = TT.Compose([
    TT.ToImage(),
    TT.ToDtype(torch.float32, scale=True),
    TT.Normalize(**spt.data.dataset_stats.ImageNet),
    TT.Resize(size=224),
])


class RealEnvMPCCCostModel(MPCCCostModel):
    def __init__(self, *args,
                 action_mean=(-0.0022, 0.0036), action_std=(0.1975, 0.1916),
                 env_damping: float = 0.35, env_resolution: int = 224,
                 phys_per_step: int = 2, n_workers: int = 12, **kwargs):
        super().__init__(*args, **kwargs)
        self._a_mean = np.asarray(action_mean, np.float64)
        self._a_std = np.asarray(action_std, np.float64)
        self._phys_per_step = int(phys_per_step)
        self._n_workers = int(os.environ.get("ORACLE_WORKERS", n_workers))
        if self._n_workers > 1:
            # spawn (not fork): the parent has CUDA initialized; forked children
            # touching the env are usually fine, but spawn is safe + workers
            # re-import only the light oracle_envpool (no torch). Pool persists.
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(self._n_workers, initializer=_ep.winit,
                                  initargs=(env_damping,))
            print(f"[RealEnvMPCC] env pool: {self._n_workers} spawn workers (damping={env_damping})", flush=True)
        else:
            self._pool = None
            _ep.winit(env_damping)                   # init module env in-process
        self._n_calls = 0

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _predict_emb(self, rollout_input: dict, physical: torch.Tensor) -> torch.Tensor:
        """Roll the REAL env for each candidate; encode → predicted_emb (B,S,T,D).

        physical: (B, S, T, A_phys),  A_phys = phys_per_step * action_block,
                  in NORMALIZED action space (un-normalize → env [-1,1]).
        state (rollout_input['state']): (B, S, T_hist, 7) expanded by CEM; the
                  env state is identical across samples → use [:, 0].
        """
        dev = next(self.world_model.parameters()).device
        B, S, T, A_phys = physical.shape
        ab = self.action_block
        pps = self._phys_per_step
        assert A_phys == pps * ab, f"A_phys {A_phys} != phys_per_step*ab {pps*ab}"

        # --- per-env physical snapshot (same for all S) ---
        st = rollout_input["state"]                  # (B,S,T_hist,7) tensor
        st = st.reshape(B, S, -1, 7)[:, 0].detach().cpu().numpy().astype(np.float64)  # (B,T_hist,7)
        cur = st[:, -1]                              # (B,7) current
        # EXACT block velocity (vx,vy,angvel) if the env exposes it (additive
        # 'block_vel' obs key) — validated to give 0.15px rollout error vs the
        # rough finite-diff path (74% vel error). Use it when present.
        if "block_vel" in rollout_input and torch.is_tensor(rollout_input["block_vel"]):
            bvk = rollout_input["block_vel"].reshape(B, S, -1, 3)[:, 0, -1].detach().cpu().numpy()
            blk_vel = bvk[:, 0:2].astype(np.float64); blk_avel = bvk[:, 2].astype(np.float64)
            dt = 0.0  # (unused; flag for the diag print)
        # finite-diff Δt self-calibrated from agent (pos & vel both observed)
        elif st.shape[1] >= 2:
            d_apos = cur[:, 0:2] - st[:, -2, 0:2]    # agent pos delta over 1 hist frame
            a_vel = cur[:, 5:7]
            num = (d_apos * a_vel).sum(1)
            den = (a_vel * a_vel).sum(1)
            dt = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)   # frames→sec scale
            dt = float(np.median(dt[np.abs(dt) > 1e-6])) if np.any(np.abs(dt) > 1e-6) else 1.0
            if not np.isfinite(dt) or abs(dt) < 1e-6:
                dt = 1.0
            blk_vel = (cur[:, 2:4] - st[:, -2, 2:4]) / dt        # (B,2) px/s
            blk_avel = (cur[:, 4] - st[:, -2, 4]) / dt           # (B,)
        else:
            blk_vel = np.zeros((B, 2)); blk_avel = np.zeros(B)

        # --- un-normalize candidate actions → env [-1,1] ---
        ph = physical.detach().cpu().numpy().astype(np.float64)          # (B,S,T,A_phys)
        ph = ph.reshape(B, S, T, ab, pps) * self._a_std + self._a_mean   # env-space
        ph = np.clip(ph, -1.0, 1.0)

        # --- roll every candidate in the env (parallel across worker envs) ---
        tasks = [(cur[b], blk_vel[b], blk_avel[b], ph[b, s])             # ph[b,s]: (T,ab,2)
                 for b in range(B) for s in range(S)]
        if self._pool is not None:
            cs = max(1, len(tasks) // (self._n_workers * 4))
            res = self._pool.map(_ep.roll_task, tasks, chunksize=cs)     # ordered (b,s)
        else:
            res = [_ep.roll_task(t) for t in tasks]
        all_imgs = []  # (B*S*T) frames in (b,s,t) order
        for fr in res:                                # fr: (T,H,W,3) uint8
            all_imgs.extend(fr[k] for k in range(fr.shape[0]))
        # batch-encode in CHUNKS (B*S*T can be ~1e4 → single batch OOMs the ViT).
        chunk = int(os.environ.get("ORACLE_ENC_CHUNK", 512))
        embs = []
        for i in range(0, len(all_imgs), chunk):
            bt = torch.stack([_TF(im) for im in all_imgs[i:i + chunk]]).to(dev)  # (c,C,224,224)
            info = {"pixels": bt[:, None]}           # (c,1,C,H,W) → encode treats t=1
            self.world_model.encode(info)
            embs.append(info["emb"][:, 0].detach())
            del bt, info
        emb = torch.cat(embs, 0)                      # (N, D)
        D = emb.shape[-1]
        post = emb.reshape(B, S, T, D).to(dev)        # (B,S,T,D) latents AFTER each horizon step

        # The WM's predicted_emb is T+1 long: [current-obs latent (anchor θ_0),
        # then 1 per horizon step]. Match it by prepending the already-encoded
        # current latent (== WM convention predicted_emb[:,0] = encoder(o_t)).
        cur_lat = rollout_input["emb"].reshape(B, S, -1, D)[:, :, -1]     # (B,S,D)
        pred_emb = torch.cat([cur_lat.unsqueeze(2), post], dim=2)         # (B,S,T+1,D)

        self._n_calls += 1
        if self._n_calls <= 3:
            _src = "EXACT(obs block_vel)" if ("block_vel" in rollout_input
                    and torch.is_tensor(rollout_input["block_vel"])) else f"finite-diff(dt={dt:.3f})"
            print(f"[RealEnvMPCC] call#{self._n_calls} B={B} S={S} T={T}(+1 anchor) ab={ab} "
                  f"vel_src={_src} blk|v|med={np.median(np.linalg.norm(blk_vel,axis=1)):.1f} "
                  f"frames={len(all_imgs)} D={D} pred_emb={tuple(pred_emb.shape)}", flush=True)
        return pred_emb


def env_res(env):
    try:
        return int(env.unwrapped.render_size)
    except Exception:
        return 224


def _set_full_state(env, s7, blk_vel, blk_avel):
    """Restore agent + block pose AND velocities (slippery coast)."""
    e = env.unwrapped if hasattr(env, "unwrapped") else env
    e._set_state(np.asarray(s7, np.float64))
    e.agent.position = (float(s7[0]), float(s7[1]))
    e.agent.velocity = (float(s7[5]), float(s7[6]))
    e.block.position = (float(s7[2]), float(s7[3]))
    e.block.angle = float(s7[4])
    e.block.velocity = (float(blk_vel[0]), float(blk_vel[1]))
    e.block.angular_velocity = float(blk_avel)
