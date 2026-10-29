"""Diffusion + Inverse-Dynamics eval (multiview cube AW).

Mirrors `eval_diffusion_mpcc.py` but replaces the MPCC/CEM inner loop with a
trained `InvDynMLP` that converts diffusion-generated latent waypoints
directly into env actions.

Pipeline per env-step:
  1. Encode the current real observation through the WM → cur_emb.
  2. Push cur_emb into the policy's stride-spaced ring buffer.
  3. If the action buffer is empty (replan boundary):
       a. Sample a fresh diffusion ref of shape (N_env, H, D).
          The diffusion sampler is configured with fix_mask[0]=1 and the
          prior's first slot is set to the last history embedding, so
          ``ref[:, 0, :] == cur_emb`` exactly (Janner inpaint anchor).
       b. For each consecutive ref pair (ref[t], ref[t+1]), invdyn
          predicts a chunk of `stride` env-actions:
            chunks = invdyn(ref[:, :-1].reshape(-1, D),
                            ref[:, 1: ].reshape(-1, D))
            chunks → (N_env, H-1, stride, A) → flatten to (N_env, (H-1)*stride, A)
       c. Take the first `action_block * receding_horizon` actions
          (= 25 for cube AW) and push into the buffer.
  4. Pop the front of the buffer and return as the env-step action.

No CEM, no ν optimization. The MPCC solver scaffolding is still built (so
`evaluate_tracking_from_dataset` can call `reset_steps`/`advance_steps` and
the dummy reference list keeps its `_references`/`_env_ref_indices`
contracts), but its `solve()` is never invoked because the policy fully
overrides `get_action`.
"""

from __future__ import annotations

import collections
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

_THIS = Path(__file__).resolve().parent
_LATENT_DIFFUSION_DIR = os.environ.get(
    "LATENT_DIFFUSION_DIR",
    str(_THIS.parent / "latent_modules" / "diffusion"),
)
sys.path.insert(0, str(_THIS))
sys.path.insert(0, _LATENT_DIFFUSION_DIR)

import stable_worldmodel as swm  # noqa: E402

from latent_diffusion.planner import (  # noqa: E402
    DiffusionPlanner,
    build_diffusion_planner,
    diffusion_generate_latent_ref_batched,
)
from eval_mpc_multiview import (  # noqa: E402
    MultiviewWorldModelPolicy,
    _setup_eval_world,
    evaluate_tracking_from_dataset,
    _attach_wrist_renderers,
    _attach_proprio_alias,
)
from eval_mpcc_multiview import _setup_eval_solver_mpcc  # noqa: E402
from latent_modules.mpcc.reference import ReferenceTrajectory  # noqa: E402
from latent_modules.invdyn import InvDynMLP  # noqa: E402


# ---------------------------------------------------------------------------
# Diffusion + InvDyn policy
# ---------------------------------------------------------------------------

class DiffusionInvDynPolicy(MultiviewWorldModelPolicy):
    """Diffusion-sampled latent ref → InvDyn action chunks. No CEM."""

    def __init__(self, *, diff: DiffusionPlanner, invdyn: InvDynMLP,
                 action_block: int, receding_horizon: int,
                 use_first_n: int | None = None,
                 dump_dir: str | None = None,
                 subsample_stride: int | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self._diff = diff
        self._invdyn = invdyn
        self._stride = int(invdyn.stride)
        self._action_block = int(action_block)
        # subsample_stride > 1 enables s=1 diffusion + s=5 InvDyn pairing
        # (Test 1, 2026-05-10): refs are subsampled at this stride BEFORE
        # forming InvDyn pairs, so InvDyn sees coarser latent deltas.
        # When set, must equal invdyn.stride; diff.stride == 1 typically.
        self._subsample_stride = int(subsample_stride) if subsample_stride else None
        # Buffer length = action_block * receding_horizon = number of env-steps
        # between diffusion replans (e.g. 5 * 5 = 25 for cube AW).
        self._buffer_max = int(action_block) * int(receding_horizon)
        self._invdyn_buffer: collections.deque = collections.deque()
        # Stride-spaced ring (mirrors DiffusionMPCCPolicy).
        self._ring_len = (diff.K - 1) * diff.stride + 1
        self._ring: torch.Tensor | None = None
        self._stride_indices = [k * diff.stride for k in range(diff.K)]
        # Optional truncation of diffusion output (e.g. H=18 ckpt → use 6).
        self._use_first_n = int(use_first_n) if use_first_n is not None else None
        # Diagnostics.
        self._n_replans = 0
        self._n_ring_updates = 0
        # Per-replan dump (cur_emb + refs) for E1 drift analysis.
        self._dump_dir = dump_dir
        if self._dump_dir is not None:
            import os
            os.makedirs(self._dump_dir, exist_ok=True)
        self._cur_emb_log: list[tuple[int, torch.Tensor]] = []
        self._env_step = 0

        if self._subsample_stride is not None:
            if self._subsample_stride != self._stride:
                raise ValueError(
                    f"subsample_stride ({self._subsample_stride}) must match "
                    f"invdyn.stride ({self._stride}) — InvDyn was trained on "
                    f"stride-spaced pairs with that gap."
                )
        elif self._stride != int(diff.stride):
            raise ValueError(
                f"invdyn.stride ({self._stride}) != diff.stride ({diff.stride}). "
                "Both must match (diffusion frame stride = invdyn chunk size). "
                "To pair fine-grained diffusion (s=1) with coarse InvDyn (s=5), "
                "set subsample_stride=invdyn.stride explicitly."
            )

    # ------------------------------------------------------------------
    def _encode_current(self, info_dict_prepared: dict) -> torch.Tensor:
        """Run the WM encoder on a curated info dict; return (N_env, D)."""
        device = next(self.solver._model.parameters()).device
        info: dict = {}
        pix = info_dict_prepared["pixels"]
        info["pixels"] = pix.to(device, non_blocking=True) if torch.is_tensor(pix) else pix
        if "wrist_pixels" in info_dict_prepared:
            wri = info_dict_prepared["wrist_pixels"]
            info["wrist_pixels"] = wri.to(device, non_blocking=True) if torch.is_tensor(wri) else wri
        enabled_views = getattr(self.solver._model, "enabled_views", ("agent", "wrist"))
        if "proprio" in enabled_views:
            for k in (getattr(self.solver._model, "proprio_keys", ()) or ()):
                if k in info_dict_prepared:
                    v = info_dict_prepared[k]
                    info[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        with torch.no_grad():
            self.solver._model.encode(info)
        emb = info["emb"]
        return emb[:, -1, :].contiguous()

    def _update_ring(self, cur_emb: torch.Tensor) -> None:
        if self._ring is None:
            self._ring = (
                cur_emb.unsqueeze(1).repeat(1, self._ring_len, 1).contiguous()
            )
        else:
            self._ring = torch.roll(self._ring, shifts=-1, dims=1)
            self._ring[:, -1, :] = cur_emb
        self._n_ring_updates += 1

    def _refs_from_ring(self) -> torch.Tensor:
        """Sample diffusion ref of shape (N_env, H_eff, D)."""
        history_K = self._ring[:, self._stride_indices, :]  # (N_env, K, D)
        refs = diffusion_generate_latent_ref_batched(
            self._diff, history_K,
            action_block=self._diff.stride,
            use_first_n=self._use_first_n,
            expand=False,                   # raw fine waypoints; no repeat_interleave
        )
        return refs                          # (N_env, H_eff, D)

    def _refs_to_actions(self, refs: torch.Tensor,
                          cur_emb: torch.Tensor | None = None) -> torch.Tensor:
        """Convert (N_env, H, D) refs → (N_env, buffer_max, A) action chunks via invdyn.

        For each consecutive pair (ref[:, t], ref[:, t+1]), invdyn yields a
        chunk of `stride` env-actions. Multi-frame InvDyn (history_K > 0):
            z_hist = [refs[t-K], refs[t-K+1], ..., refs[t]] with replicate-pad
            at refs[0] for indices < 0. Matches stride-spaced training-time
            history (training uses latents at idx_t-k*stride, k=0..K).
        We flatten across pairs and take the first `buffer_max` actions. Any
        actions beyond buffer_max are discarded — the next replan re-anchors.

        cur_emb (N_env, D), if provided, replaces refs[0] (the diffusion anchor
        slot) in the FIRST pair only — InvDyn sees (real cur_emb, diffusion
        refs[1]) → action. Matches training-time (real, real) distribution more
        precisely than (refs[0]≈anchor+DDIM noise, refs[1]). Agent-3 fix
        2026-05-10. Subsequent pairs (refs[1..H-2], refs[2..H-1]) unchanged.
        """
        N_env, H, D = refs.shape
        if H < 2:
            raise RuntimeError(
                f"diffusion ref has H={H} (<2); cannot form pairs for invdyn."
            )
        K_inv = int(getattr(self._invdyn, "history_K", 0))
        device = next(self._invdyn.parameters()).device
        refs = refs.to(device, non_blocking=True)

        # Optional subsample: when fine-grained s=1 diffusion is paired with
        # coarse s=5 InvDyn, take every Nth ref so InvDyn pairs span larger
        # latent deltas (better SNR vs noise floor). For s=1 + s=5 InvDyn:
        # subsample_stride=5 → refs[0,5,10,...,80] (17 frames from 83).
        if self._subsample_stride is not None and self._subsample_stride > 1:
            refs = refs[:, ::self._subsample_stride, :].contiguous()
            H = refs.shape[1]
            if H < 2:
                raise RuntimeError(
                    f"after subsample(stride={self._subsample_stride}) only H={H} "
                    f"frames left; need >= 2."
                )

        z_tp = refs[:, 1:, :].reshape(-1, D)            # (N_env*(H-1), D)
        if K_inv == 0:
            z_t = refs[:, :-1, :].clone()                # (N_env, H-1, D)
            if cur_emb is not None:
                # Replace anchor slot (first pair's z_t) with real cur_emb.
                z_t[:, 0, :] = cur_emb.to(z_t.device).to(z_t.dtype)
            z_t = z_t.reshape(-1, D)
            with torch.no_grad():
                chunks = self._invdyn(z_t, z_tp)        # legacy path
        else:
            # Build stride-spaced history per pair t: indices clamp(t-K, 0), ..., t.
            # Result: (N_env, H-1, K+1, D) → flatten to (N_env*(H-1), K+1, D).
            # Use index gather to replicate-pad at refs[0] for t < K.
            t_idx = torch.arange(H - 1, device=device)  # (H-1,) anchors at t = 0..H-2
            offsets = torch.arange(K_inv, -1, -1, device=device)  # [K, K-1, ..., 1, 0]
            # k=0 of K1 -> offset K (lookback K), k=K -> offset 0
            hist_idx = (t_idx.unsqueeze(1) - offsets.unsqueeze(0)).clamp(min=0)  # (H-1, K+1)
            # Gather: (N_env, H-1, K+1, D)
            z_hist = refs[:, hist_idx, :]
            z_hist = z_hist.reshape(-1, K_inv + 1, D)   # (N_env*(H-1), K+1, D)
            with torch.no_grad():
                chunks = self._invdyn(z_hist, z_tp)     # (N_env*(H-1), stride, A)
        a_dim = int(chunks.shape[-1])
        chunks = chunks.view(N_env, H - 1, self._stride, a_dim)
        actions = chunks.reshape(N_env, (H - 1) * self._stride, a_dim)
        if actions.shape[1] < self._buffer_max:
            raise RuntimeError(
                f"diffusion ref produced only {actions.shape[1]} env-actions but "
                f"buffer_max={self._buffer_max} required. Increase H "
                f"(currently {H}, need >= {self._buffer_max // self._stride + 1})."
            )
        return actions[:, :self._buffer_max, :].contiguous()

    # ------------------------------------------------------------------
    def get_action(self, info_dict, **kwargs):
        # Run _prepare_info on a MINIMAL view of info_dict — only the keys
        # the WM encoder actually consumes (pixels, optional wrist_pixels +
        # proprio). Skipping the 'action' key avoids running the sklearn
        # action-scaler on the policy's past-action buffer (which has a
        # 3-d shape that breaks `StandardScaler.transform` on this branch).
        enabled_views = getattr(self.solver._model, "enabled_views", ("agent", "wrist"))
        proprio_keys = tuple(getattr(self.solver._model, "proprio_keys", ()) or ())
        needed = {"pixels"}
        if "wrist" in enabled_views:
            needed.add("wrist_pixels")
        if "proprio" in enabled_views:
            needed.update(proprio_keys)
        encode_info = {k: info_dict[k] for k in needed if k in info_dict}
        prepared = self._prepare_info(encode_info)
        cur_emb = self._encode_current(prepared)
        self._update_ring(cur_emb)
        if self._dump_dir is not None:
            self._cur_emb_log.append((self._env_step, cur_emb.detach().cpu().clone()))
        self._env_step += 1

        if len(self._invdyn_buffer) == 0:
            refs = self._refs_from_ring()                 # (N_env, H_eff, D)
            if self._dump_dir is not None:
                import os
                torch.save({
                    "replan_idx": self._n_replans,
                    "env_step_at_replan": self._env_step - 1,
                    "ring": self._ring.detach().cpu(),
                    "history_K_per_env": [
                        self._ring[:, self._stride_indices, :][i].detach().cpu()
                        for i in range(self._ring.shape[0])
                    ],
                    "refs_per_env": [refs[i].detach().cpu() for i in range(refs.shape[0])],
                    "cur_emb_log": list(self._cur_emb_log),
                }, os.path.join(self._dump_dir, f"replan_{self._n_replans:03d}.pt"))
            actions = self._refs_to_actions(refs, cur_emb=cur_emb)         # (N_env, buffer_max, A)
            for t in range(actions.shape[1]):
                self._invdyn_buffer.append(actions[:, t, :].cpu())
            self._n_replans += 1

        return self._invdyn_buffer.popleft()              # (N_env, A)


# ---------------------------------------------------------------------------
# CleanDiffuser-trained ckpt variant — same ring buffer + InvDyn loop,
# different sample fn. Activated automatically when ckpt has sibling
# `meta.json` (CleanDiffuser format). See `latent_diffusion_cd/planner.py`.
# ---------------------------------------------------------------------------

class CleanDiffuserDiffInvDynPolicy(DiffusionInvDynPolicy):
    """Drop-in for DiffusionInvDynPolicy that uses CleanDiffuser sample fn.

    Overrides only `_refs_from_ring`. The `diff` injected via constructor is
    a CleanDiffuserDiffPlanner (duck-typed: has K, stride, H, obs_dim — same
    interface DiffusionInvDynPolicy reads).
    """

    def _refs_from_ring(self) -> torch.Tensor:
        # Local import: latent_diffusion_cd.planner imports cleandiffuser.*,
        # only available when the eval sbatch has set the PYTHONPATH.
        from latent_diffusion_cd.planner import cleandiffuser_generate_latent_ref_batched

        history_K = self._ring[:, self._stride_indices, :]
        refs = cleandiffuser_generate_latent_ref_batched(
            self._diff, history_K,
            action_block=self._diff.stride,
            use_first_n=self._use_first_n,
            expand=False,
        )
        return refs                          # (N_env, H_eff, D)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_invdyn(ckpt_path: str | Path, device: str) -> InvDynMLP:
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ck.get("config") or {}
    args_dict = ck.get("args", {})
    stride = int(cfg.get("stride", args_dict.get("stride", 1)))
    hidden = int(cfg.get("hidden", args_dict.get("hidden", 512)))
    add_norm = bool(cfg.get("add_norm", not args_dict.get("no_norm", False)))
    dropout = float(cfg.get("dropout", args_dict.get("dropout", 0.0)))
    z_dim = int(cfg.get("z_dim"))
    a_dim = int(cfg.get("a_dim"))
    history_K = int(cfg.get("history_K", args_dict.get("history_K", 0)))
    model = InvDynMLP(z_dim=z_dim, a_dim=a_dim, stride=stride, hidden=hidden,
                      add_norm=add_norm, dropout=dropout,
                      history_K=history_K).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"[invdyn] loaded {ckpt_path}  step={ck.get('step', '?')}  "
          f"val_mse={ck.get('val_mse', '?')}  stride={stride}  z_dim={z_dim}  "
          f"history_K={history_K}")
    return model


@hydra.main(version_base=None, config_path="./config/eval", config_name="multiview_cube_mpcc")
def run(cfg: DictConfig):
    """Run multi-view diffusion+InvDyn evaluation."""
    diff_cfg = cfg.get("diffusion", None)
    assert diff_cfg is not None and diff_cfg.get("ckpt_path"), (
        "cfg.diffusion.ckpt_path required "
        "(Hydra: +diffusion.ckpt_path=/path/to/planner.pt)"
    )
    assert diff_cfg.get("invdyn_ckpt"), (
        "cfg.diffusion.invdyn_ckpt required "
        "(Hydra: +diffusion.invdyn_ckpt=/path/to/best.pt)"
    )

    # -- Phase 1: env + dataset + transforms (shared with GT-MPCC) -----------
    eval_world = _setup_eval_world(cfg)

    # -- Phase 2: build diffusion planner BEFORE invdyn. Auto-dispatch between
    # legacy format (VelLossDiffusionSDE bundle in ckpt meta) and CleanDiffuser
    # format (sibling meta.json + normalizer_stats.npz next to ckpt).
    overrides = (OmegaConf.to_container(diff_cfg.overrides, resolve=True)
                 if diff_cfg.get("overrides") else None)
    from latent_diffusion_cd.planner import is_cleandiffuser_ckpt
    _is_cd = is_cleandiffuser_ckpt(str(diff_cfg.ckpt_path))
    print(f"[diff+InvDyn eval] ckpt format = "
          f"{'CleanDiffuser (meta.json sibling)' if _is_cd else 'legacy (VelLossDiffusionSDE)'}")
    if _is_cd:
        from latent_diffusion_cd.planner import build_cleandiffuser_diff_planner
        diff = build_cleandiffuser_diff_planner(
            ckpt_path=str(diff_cfg.ckpt_path), device="cuda",
        )
        _policy_cls = CleanDiffuserDiffInvDynPolicy
    else:
        diff = build_diffusion_planner(
            diff_cfg.ckpt_path, device="cuda", overrides=overrides,
        )
        _policy_cls = DiffusionInvDynPolicy
    plan_action_block = int(cfg.plan_config.action_block)
    receding_horizon = int(cfg.plan_config.receding_horizon)
    # subsample_stride: pair fine-grained s=1 diffusion with coarse s=5 InvDyn.
    # When set, refs[::subsample_stride] are used as InvDyn pairs (better SNR).
    subsample_stride = diff_cfg.get("subsample_stride", None)
    subsample_stride = int(subsample_stride) if subsample_stride else None

    invdyn = _load_invdyn(diff_cfg.invdyn_ckpt, device="cuda")
    if subsample_stride is not None:
        # Subsample mode: action_block must match invdyn.stride (NOT diff.stride).
        assert plan_action_block == invdyn.stride, (
            f"subsample mode: plan_action_block ({plan_action_block}) must match "
            f"invdyn.stride ({invdyn.stride})."
        )
        assert subsample_stride == invdyn.stride, (
            f"subsample_stride ({subsample_stride}) must equal invdyn.stride "
            f"({invdyn.stride})."
        )
    else:
        assert diff.stride == plan_action_block, (
            f"diffusion.stride ({diff.stride}) != plan_config.action_block "
            f"({plan_action_block})."
        )
        if invdyn.stride != diff.stride:
            raise RuntimeError(
                f"invdyn stride={invdyn.stride} != diff stride={diff.stride}. "
                "Train the invdyn ckpt with --stride matching diffusion frame stride."
            )
    if invdyn.z_dim != diff.obs_dim:
        raise RuntimeError(
            f"invdyn z_dim={invdyn.z_dim} != diff obs_dim={diff.obs_dim}. "
            "Latent dimension mismatch — invdyn was trained on a different "
            "WM encoder."
        )

    use_first_n = diff_cfg.get("use_first_n", None)
    effective_H = int(use_first_n) if use_first_n is not None else diff.H
    if subsample_stride is not None:
        # After subsample, H_effective = ceil(effective_H / subsample_stride).
        # Each pair → invdyn.stride env-actions.
        H_after_subsample = (effective_H + subsample_stride - 1) // subsample_stride
        n_actions_available = (H_after_subsample - 1) * invdyn.stride
    else:
        n_actions_available = (effective_H - 1) * diff.stride
    n_actions_needed = plan_action_block * receding_horizon
    if n_actions_available < n_actions_needed:
        raise ValueError(
            f"Diffusion produces {n_actions_available} env-actions per replan "
            f"but action_block × receding_horizon = {n_actions_needed} are needed. "
            "Increase H or reduce receding_horizon."
        )

    # -- Phase 3: solver + DiffusionInvDynPolicy ---------------------------
    # Build the standard MPCC solver via _setup_eval_solver_mpcc with our
    # custom policy_cls. The solver's solve() is never called by the policy,
    # but reset_steps/advance_steps + _references/_env_ref_indices are still
    # touched by evaluate_tracking_from_dataset.
    n_eval = eval_world.n_eval
    D = diff.obs_dim
    dummy_emb = torch.zeros(2, D)                # min N=2 for arc-length
    dummy_ref = ReferenceTrajectory.from_latents(dummy_emb)

    # Pre-resolve dump_dir for per-replan ref/cur_emb logging (E1 drift analysis).
    _policy_dir_pre = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
    _run_name_pre = cfg.output.get("run_name") or "eval_diffusion_invdyn"
    _dump_dir_pre = str(_policy_dir_pre / _run_name_pre / "ref_dumps")

    model, solver, policy, plan_config = _setup_eval_solver_mpcc(
        cfg, eval_world,
        gt_refs=False,
        policy_cls=_policy_cls,                # ← auto-dispatched above
        policy_kwargs={
            "diff": diff,
            "invdyn": invdyn,
            "action_block": plan_action_block,
            "receding_horizon": receding_horizon,
            "use_first_n": use_first_n,
            "dump_dir": _dump_dir_pre,
            "subsample_stride": subsample_stride,
        },
    )

    # Standard cube WM patch: HF ViT loaded with this transformers version
    # (4.46.x) lacks `dropout` submodule on `ViTSelfAttention` and is missing
    # some `ViTConfig` fields. Mirror the patch used elsewhere
    # (cache_latents_perview_aw.py, validate_diffusion_planner.py, train_decoder.py).
    import transformers as _hf_transformers
    for _m in model.modules():
        if hasattr(_m, "config") and isinstance(_m.config, _hf_transformers.PretrainedConfig):
            _fresh = type(_m.config)()
            for _k, _v in _fresh.__dict__.items():
                if _k not in _m.config.__dict__:
                    setattr(_m.config, _k, _v)
        if "ViTSelfAttention" in type(_m).__name__ and not hasattr(_m, "dropout"):
            _m.dropout = torch.nn.Dropout(getattr(_m, "dropout_prob", 0.0))

    # Pre-populate solver._references with dummy refs so cost_model.__init__
    # has shape. The DiffusionInvDynPolicy never refreshes these (it bypasses
    # the solver entirely), but evaluate_tracking_from_dataset reads them.
    solver._references = [
        ReferenceTrajectory.from_latents(dummy_emb.clone()) for _ in range(n_eval)
    ]
    solver._env_ref_indices = np.arange(n_eval)

    # -- Phase 4: bind policy to world (triggers solver.configure) ----------
    eval_world.world.set_policy(policy)
    enabled_views = getattr(model, "enabled_views", ("agent", "wrist"))
    proprio_keys = tuple(getattr(model, "proprio_keys", ()) or ())
    if "wrist" in enabled_views:
        _attach_wrist_renderers(eval_world.world, img_size=cfg.world.get("width", 224))
        print(f"[renderer] wrist renderer attached (enabled_views={list(enabled_views)})")
    if "proprio" in enabled_views:
        _attach_proprio_alias(eval_world.world, proprio_keys)

    # -- Phase 5: output path -----------------------------------------------
    policy_dir = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
    run_name = cfg.output.get("run_name") or "eval_diffusion_invdyn"
    results_path = policy_dir / run_name
    results_path.mkdir(parents=True, exist_ok=True)
    print(f"[diff+InvDyn eval] episodes={np.unique(eval_world.eval_episodes).tolist()}, "
          f"goal_offset={eval_world.goal_offset}, eval_budget={eval_world.eval_budget}, "
          f"n={n_eval}, diff_ckpt={Path(diff_cfg.ckpt_path).name}, "
          f"invdyn_ckpt={Path(diff_cfg.invdyn_ckpt).name}")
    print(f"[diff+InvDyn eval] H_eff={effective_H}  stride={diff.stride}  "
          f"actions/replan={n_actions_needed}  available={n_actions_available}")

    # -- Phase 6: run eval (reuse evaluate_tracking_from_dataset) ----------
    callables = (OmegaConf.to_container(cfg.eval.callables, resolve=True)
                 if cfg.eval.get("callables") else None)

    # Build RolloutLogger (opt-in via +eval.rollout_log=true).
    rollout_logger = None
    if cfg.eval.get("rollout_log", False):
        from latent_modules.eval import RolloutLogger
        method_name = cfg.eval.get("rollout_method_name", "dtg-invdyn")
        task_name = cfg.eval.get("rollout_task_name", "cube-mv")
        rollout_logger = RolloutLogger(
            method=method_name,
            task=task_name,
            out_dir=results_path,
            n_envs=eval_world.n_eval,
            eval_budget=eval_world.eval_budget,
        )
        rollout_logger.set_episodes(
            episode_idx=eval_world.eval_episodes.tolist(),
            start_step=eval_world.eval_start_idx.tolist(),
            goal_offset=(eval_world.goal_offsets.tolist()
                         if hasattr(eval_world.goal_offsets, "tolist")
                         else [int(eval_world.goal_offsets)] * eval_world.n_eval),
        )
        rollout_logger.add_meta(
            policy=cfg.policy,
            eval_dataset=cfg.eval.dataset_name,
            train_dataset=cfg.dataset.get("stats", cfg.eval.dataset_name),
            cache_dir=cfg.get("cache_dir", None),
            seed=int(cfg.seed),
            action_block=int(cfg.plan_config.action_block),
            horizon=int(cfg.plan_config.horizon),
            receding_horizon=int(cfg.plan_config.receding_horizon),
            plan_config=cfg.plan_config,
            world=cfg.world,
            diffusion_ckpt=str(diff_cfg.ckpt_path),
            invdyn_ckpt=str(diff_cfg.invdyn_ckpt),
            diffusion_arch={"H": int(diff.H), "K": int(diff.K),
                            "stride": int(diff.stride), "D": int(diff.obs_dim)},
            H_eff=int(effective_H),
            n_actions_per_replan=int(n_actions_needed),
            start_offset=cfg.eval.get("start_offset", None),
        )
        rollout_logger.attach_demo_ttc(cfg.eval.get("demo_ttc_path", None))
        print(f"[rollout_log] enabled — writing to {results_path}")

    t0 = time.time()
    metrics = evaluate_tracking_from_dataset(
        world=eval_world.world,
        dataset=eval_world.dataset,
        solver=solver,
        episodes_idx=eval_world.eval_episodes.tolist(),
        start_steps=eval_world.eval_start_idx.tolist(),
        goal_offset_steps=(eval_world.goal_offsets.tolist()
                           if hasattr(eval_world.goal_offsets, "tolist")
                           else eval_world.goal_offsets),
        eval_budget=eval_world.eval_budget,
        callables=callables,
        video_path=str(results_path),
        save_video=cfg.eval.get("save_video", True),
        debug=False,                          # debug video relies on CEM internals
        model=None,
        logger=rollout_logger,
    )
    elapsed = time.time() - t0
    print(f"[diff+InvDyn eval] done in {elapsed:.1f}s  "
          f"replans={policy._n_replans}  ring_updates={policy._n_ring_updates}")

    # -- Phase 7: save results --------------------------------------------
    behavioral = metrics.get("behavioral_aggregate", {})
    success_rate = metrics.get("success_rate", float("nan"))
    print(f"[diff+InvDyn eval] success_rate={success_rate:.1f}%  behavioral={behavioral}")

    # Per-env first_success_step (TTF) for downstream compare_solvers.py.
    fs = metrics.get("first_success_step")
    es = metrics.get("episode_successes")
    out_npz = results_path / "diffusion_invdyn_results.npz"
    np.savez(
        out_npz,
        success_rate=success_rate,
        n_replans=policy._n_replans,
        n_ring_updates=policy._n_ring_updates,
        elapsed=elapsed,
        diff_ckpt_path=str(diff_cfg.ckpt_path),
        invdyn_ckpt_path=str(diff_cfg.invdyn_ckpt),
        first_success_step=(np.asarray(fs) if fs is not None
                            else np.zeros(0, dtype=np.int64)),
        episode_successes=(np.asarray(es) if es is not None
                           else np.zeros(0, dtype=bool)),
    )
    out_txt = results_path / "diffusion_invdyn_results.txt"
    out_txt.write_text(
        f"diff_ckpt      : {diff_cfg.ckpt_path}\n"
        f"invdyn_ckpt    : {diff_cfg.invdyn_ckpt}\n"
        f"meta arch      : H={diff.H} K={diff.K} S={diff.stride} D={diff.obs_dim}\n"
        f"H_eff (used)   : {effective_H}\n"
        f"actions/replan : {n_actions_needed} (available {n_actions_available})\n"
        f"sample         : {diff.sample_steps}/{diff.solver}  "
        f"w_cfg={diff.w_cfg}\n"
        f"start_offset   : {cfg.eval.get('start_offset', None)}\n"
        f"replans        : {policy._n_replans}\n"
        f"ring_updates   : {policy._n_ring_updates}\n"
        f"success_rate   : {success_rate:.2f}%\n"
        f"behavioral     : {behavioral}\n"
        f"elapsed (s)    : {elapsed:.1f}\n"
        + (
            "\n# per-env outcomes (TTF=first_success_step; -1=fail)\n"
            + "".join(
                f"env_{i}: success={bool(es[i])}, first_success={int(fs[i])}\n"
                for i in range(len(es))
            )
            if (fs is not None and es is not None) else ""
        )
    )
    print(f"[diff+InvDyn eval] saved {out_npz}  +  {out_txt}")


if __name__ == "__main__":
    run()
