"""Diffusion-MPCC eval (multiview-only, slim rewrite).

Mirrors `eval_diffusion_mpc.py` pattern: reuses `_setup_eval_world` +
`_setup_eval_solver_mpcc(gt_refs=False, ...)` from `eval_mpcc_multiview.py`
(env, dataset, transforms, solver, policy build), wraps the policy in
`DiffusionMPCCPolicy` to add ring buffer + per-replan diffusion ref refresh.

Differences from diffusion-MPC:
  - Refs are `ReferenceTrajectory` objects (arc-length parametrized) instead of
    raw (T, D) tensors. Built via `ReferenceTrajectory.from_latents(...)`.
  - Policy calls `solver.refresh_reference_trajectories(...)` (delegates to
    cost_model.refresh_reference_trajectories) instead of
    `solver.refresh_references(...)`.
  - Diffusion output is NOT repeat_interleaved — fine waypoints (H frames) are
    used directly. Arc-length parametrization in ReferenceTrajectory absorbs
    the stride; nu_scale formula `action_block / (c · stride · ref.N)` handles
    the env-step-vs-frame density (set `+solver.reference.stride=5` for
    diffusion at frame stride 5).
  - Output actions are physical-only (ν stripped inside EnsembleMPCCSolver
    or MPCCSolver). Policy never sees ν.

Hydra entry. Inherits `multiview_cube_mpcc` config defaults; expects diffusion
ckpt path under `cfg.diffusion.ckpt_path`. Optional inference overrides under
`cfg.diffusion.{sample_steps, w_cfg, x_clip_abs, n_ensemble, dump_dir,
duplicate_current_history, use_first_n}` — defaults read from ckpt meta.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Repo bootstrap (mirrors eval_diffusion_mpc.py)
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
    _setup_eval_world,
    evaluate_tracking_from_dataset,
    _attach_wrist_renderers,
    _attach_proprio_alias,
)
from eval_mpcc_multiview import (  # noqa: E402
    MultiviewWorldModelPolicy,
    _setup_eval_solver_mpcc,
)
from latent_modules.mpcc.reference import ReferenceTrajectory  # noqa: E402


# ---------------------------------------------------------------------------
# Diffusion-aware policy
# ---------------------------------------------------------------------------

class DiffusionMPCCPolicy(MultiviewWorldModelPolicy):
    """Per-env-step ring buffer + diffusion-ref refresh on each replan.

    Mirrors `DiffusionMPCPolicy` (in eval_diffusion_mpc.py) but for MPCC:
    refs are wrapped as `ReferenceTrajectory` (arc-length [0, 1]) and pushed
    via `solver.refresh_reference_trajectories(...)`.
    """

    def __init__(self, *, diff: DiffusionPlanner, action_block: int,
                 dump_dir: str | None = None,
                 use_first_n: int | None = None,
                 n_ensemble: int = 1,
                 duplicate_current_history: bool = False,
                 view_dims: list | tuple | None = None,
                 view_weights: list | tuple | None = None,
                 interpolation: str = "linear",
                 **kwargs):
        super().__init__(**kwargs)
        self._diff = diff
        self._diff_action_block = int(action_block)
        self._view_dims = view_dims
        self._view_weights = view_weights
        self._interpolation = interpolation
        # ring length = (K-1)*stride + 1 = covers history from t-(K-1)*s to t
        self._ring_len = (diff.K - 1) * diff.stride + 1
        self._ring: torch.Tensor | None = None         # (N_env, ring_len, D), lazy
        self._stride_indices = [k * diff.stride for k in range(diff.K)]
        # Diagnostic counters
        self._n_replans = 0
        self._n_ring_updates = 0
        # Per-step cur_emb log (one entry per env-step); saved alongside replan
        # dumps for offline drift analysis.
        self._cur_emb_log: list[tuple[int, torch.Tensor]] = []
        # Optional truncation of diffusion output (e.g. H=18 ckpt → first M).
        self._use_first_n = int(use_first_n) if use_first_n is not None else None
        # Ensemble: N candidate refs per env per replan.
        self._n_ensemble = max(1, int(n_ensemble))
        # Diagnostic: replace K-frame strided history with K copies of current
        # obs (ring[-1]). Same toggle as DiffusionMPCPolicy.
        self._duplicate_current_history = bool(duplicate_current_history)
        # Optional ref dump per replan (for debugging mode-collapse).
        self._dump_dir = dump_dir
        if dump_dir is not None:
            import os
            os.makedirs(dump_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _encode_current(self, info_dict_prepared: dict) -> torch.Tensor:
        """Build a CURATED info dict and run `PerViewJEPA.encode` to get the
        latent for the current frame. Mirrors `DiffusionMPCPolicy._encode_current`
        — same `info["action"]` exclusion to avoid action_encoder shape mismatch.
        """
        device = next(self.solver._model.parameters()).device
        info: dict = {}
        pix = info_dict_prepared["pixels"]
        info["pixels"] = pix.to(device, non_blocking=True) if torch.is_tensor(pix) else pix
        if "wrist_pixels" in info_dict_prepared:
            wri = info_dict_prepared["wrist_pixels"]
            info["wrist_pixels"] = wri.to(device, non_blocking=True) if torch.is_tensor(wri) else wri
        # Proprio keys (if enabled).
        enabled_views = getattr(self.solver._model, "enabled_views", ("agent", "wrist"))
        if "proprio" in enabled_views:
            proprio_keys = getattr(self.solver._model, "proprio_keys", ()) or ()
            for k in proprio_keys:
                if k in info_dict_prepared:
                    v = info_dict_prepared[k]
                    info[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        with torch.no_grad():
            self.solver._model.encode(info)
        emb = info["emb"]                 # (N, T, D)
        return emb[:, -1, :].contiguous() # (N, D)

    def _update_ring(self, cur_emb: torch.Tensor) -> None:
        if self._ring is None:
            N = cur_emb.shape[0]
            self._ring = (
                cur_emb.unsqueeze(1)
                .repeat(1, self._ring_len, 1)
                .contiguous()
            )
        else:
            self._ring = torch.roll(self._ring, shifts=-1, dims=1)
            self._ring[:, -1, :] = cur_emb
        if self._dump_dir is not None:
            env_step = self._n_ring_updates
            self._cur_emb_log.append((int(env_step), cur_emb.detach().cpu().clone()))
        self._n_ring_updates += 1

    def _refs_from_ring(self) -> list[ReferenceTrajectory]:
        """Sample diffusion refs from ring + wrap as ReferenceTrajectory.

        Returns N_env * N_ensemble refs interleaved per env so the solver's
        repeat_interleave expansion of info_dict batch dim aligns with these
        (row i*E + e maps to env i, candidate e).

        Note vs diffusion-MPC: NO `repeat_interleave(action_block)` on raw
        diffusion output — MPCC uses arc-length on fine waypoints. The
        nu_scale formula already absorbs stride.
        """
        N_env = self._ring.shape[0]
        E = self._n_ensemble
        if self._duplicate_current_history:
            cur = self._ring[:, -1:, :]                                  # (N_env, 1, D)
            history_K_per_env = cur.expand(-1, self._diff.K, -1).contiguous()
        else:
            history_K_per_env = self._ring[:, self._stride_indices, :]  # (N_env, K, D)
        history_K_batch = history_K_per_env.repeat_interleave(E, dim=0)  # (N_env*E, K, D)
        # Pass expand=False to skip both stride assertion and
        # repeat_interleave inside the planner — MPCC's arc-length
        # parametrization needs fine waypoints (no duplicate adjacents).
        # action_block param is unused when expand=False, but pass nominal
        # value for clarity.
        refs_batched = diffusion_generate_latent_ref_batched(
            self._diff, history_K_batch,
            action_block=self._diff_action_block,
            use_first_n=self._use_first_n,
            expand=False,
        )  # (N_env*E, H_eff, D) — fine, no repeat
        # Wrap each row as a ReferenceTrajectory (computes arc-length theta).
        refs: list[ReferenceTrajectory] = [
            ReferenceTrajectory.from_latents(
                refs_batched[i],
                view_dims=self._view_dims,
                view_weights=self._view_weights,
                interpolation=self._interpolation,
            )
            for i in range(N_env * E)
        ]
        # Optional dump per replan.
        if self._dump_dir is not None:
            import os
            torch.save(
                {
                    "replan_idx": self._n_replans,
                    "n_ensemble": E,
                    "ring": self._ring.detach().cpu(),
                    "stride_indices": list(self._stride_indices),
                    "history_K_per_env": [
                        history_K_per_env[i].detach().cpu()
                        for i in range(N_env)
                    ],
                    "cond_mode": "duplicate_current" if self._duplicate_current_history else "stride",
                    "refs_per_env_embeddings": [r.embeddings.detach().cpu() for r in refs],
                    "refs_per_env_theta": [r.theta.detach().cpu() for r in refs],
                    "n_ring_updates": int(self._n_ring_updates),
                    "cur_emb_log": list(self._cur_emb_log),
                },
                os.path.join(self._dump_dir, f"replan_{self._n_replans:03d}.pt"),
            )
        return refs

    # ------------------------------------------------------------------
    def get_action(self, info_dict, **kwargs):
        # Run prepare_info on a shallow copy for our own encode.
        prepared = self._prepare_info(dict(info_dict))
        cur_emb = self._encode_current(prepared)
        self._update_ring(cur_emb)

        # Refresh diffusion ref BEFORE super calls solver.
        if len(self._action_buffer) == 0:
            new_refs = self._refs_from_ring()
            self.solver.refresh_reference_trajectories(new_refs)
            self._n_replans += 1

        return super().get_action(info_dict, **kwargs)


# ---------------------------------------------------------------------------
# CleanDiffuser-trained ckpt variant — same ring buffer, different sample fn.
# Activated automatically when ckpt has sibling `meta.json` (CleanDiffuser
# format). See `latent_diffusion_cd/planner.py` for the bundle + builder +
# sample fn.
# ---------------------------------------------------------------------------

class CleanDiffuserDiffMPCCPolicy(DiffusionMPCCPolicy):
    """Drop-in for DiffusionMPCCPolicy that uses CleanDiffuser sample fn.

    Overrides only `_refs_from_ring`. Reuses `_update_ring`, `_encode_current`,
    `get_action`, and all init logic from the parent. The `diff` injected via
    constructor is a CleanDiffuserDiffPlanner (duck-typed: has K, stride, H,
    obs_dim — same interface DiffusionMPCCPolicy reads).
    """

    def _refs_from_ring(self) -> list[ReferenceTrajectory]:
        # Local import: latent_diffusion_cd.planner imports cleandiffuser.*,
        # only available when the eval sbatch has set the PYTHONPATH.
        from latent_diffusion_cd.planner import cleandiffuser_generate_latent_ref_batched

        N_env = self._ring.shape[0]
        E = self._n_ensemble
        if self._duplicate_current_history:
            cur = self._ring[:, -1:, :]
            history_K_per_env = cur.expand(-1, self._diff.K, -1).contiguous()
        else:
            history_K_per_env = self._ring[:, self._stride_indices, :]
        history_K_batch = history_K_per_env.repeat_interleave(E, dim=0)

        refs_batched = cleandiffuser_generate_latent_ref_batched(
            self._diff, history_K_batch,
            action_block=self._diff_action_block,
            use_first_n=self._use_first_n,
            expand=False,
        )    # (N_env*E, H_eff, D)

        refs: list[ReferenceTrajectory] = [
            ReferenceTrajectory.from_latents(
                refs_batched[i],
                view_dims=self._view_dims,
                view_weights=self._view_weights,
                interpolation=self._interpolation,
            )
            for i in range(N_env * E)
        ]
        if self._dump_dir is not None:
            import os as _os
            torch.save(
                {
                    "replan_idx": self._n_replans,
                    "n_ensemble": E,
                    "ring": self._ring.detach().cpu(),
                    "stride_indices": list(self._stride_indices),
                    "history_K_per_env": [
                        history_K_per_env[i].detach().cpu() for i in range(N_env)
                    ],
                    "cond_mode": "duplicate_current" if self._duplicate_current_history else "stride",
                    "refs_per_env_embeddings": [r.embeddings.detach().cpu() for r in refs],
                    "refs_per_env_theta": [r.theta.detach().cpu() for r in refs],
                    "n_ring_updates": int(self._n_ring_updates),
                    "ckpt_format": "cleandiffuser",
                },
                _os.path.join(self._dump_dir, f"replan_{self._n_replans:03d}.pt"),
            )
        return refs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./config/eval", config_name="multiview_cube_mpcc")
def run(cfg: DictConfig):
    """Run multi-view diffusion-MPCC evaluation."""
    print(f"Policy: {cfg.policy}")
    diff_cfg = cfg.get("diffusion", None)
    assert diff_cfg is not None and diff_cfg.get("ckpt_path"), (
        "cfg.diffusion.ckpt_path required "
        "(Hydra: +diffusion.ckpt_path=/path/to/planner.pt)"
    )

    # -- Phase 1: env + dataset + transforms (shared with GT-MPCC) -----------
    eval_world = _setup_eval_world(cfg)

    # -- Phase 2: build diffusion planner BEFORE solver. Auto-dispatch between
    # legacy format (VelLossDiffusionSDE bundle in ckpt meta) and CleanDiffuser
    # format (sibling meta.json + normalizer_stats.npz next to ckpt).
    overrides = (OmegaConf.to_container(diff_cfg.overrides, resolve=True)
                 if diff_cfg.get("overrides") else None)
    from latent_diffusion_cd.planner import is_cleandiffuser_ckpt
    _is_cd = is_cleandiffuser_ckpt(str(diff_cfg.ckpt_path))
    print(f"[diff-MPCC eval] ckpt format = "
          f"{'CleanDiffuser (meta.json sibling)' if _is_cd else 'legacy (VelLossDiffusionSDE)'}")
    if _is_cd:
        from latent_diffusion_cd.planner import build_cleandiffuser_diff_planner
        diff = build_cleandiffuser_diff_planner(
            ckpt_path=str(diff_cfg.ckpt_path), device="cuda",
        )
        _policy_cls = CleanDiffuserDiffMPCCPolicy
    else:
        diff = build_diffusion_planner(
            diff_cfg.ckpt_path, device="cuda", overrides=overrides,
        )
        _policy_cls = DiffusionMPCCPolicy
    plan_action_block = int(cfg.plan_config.action_block)
    # Note: diff.stride and plan_action_block are independent dimensions —
    # plan_action_block must match WM frame_skip (action_encoder Conv1d ch_in)
    # while diff.stride controls reference frame density. nu_scale formula
    # (action_block / (c · stride · ref.N)) absorbs both → ν=1.0 = natural
    # pace regardless of (action_block, stride) combination.
    if diff.stride != plan_action_block:
        print(f"[diff-MPCC eval] diff.stride={diff.stride} != "
              f"plan_action_block={plan_action_block}. nu_scale handles the "
              f"decoupling — proceeding.")

    # MPCC uses arc-length, so horizon T (cost.py) is decoupled from diff.H.
    # use_first_n optionally truncates diffusion output (e.g., H=18 ckpt with
    # use_first_n=6 to test shorter refs).
    use_first_n = diff_cfg.get("use_first_n", None) if diff_cfg else None
    effective_H = int(use_first_n) if use_first_n is not None else diff.H
    if effective_H < 2:
        raise ValueError(
            f"effective diffusion H ({effective_H}) < 2. ReferenceTrajectory "
            "requires at least 2 waypoints for arc-length parametrization."
        )

    # Auto-set reference.stride to diff.stride if not user-overridden. Cost
    # model formula: nu_scale = action_block / (c · stride · ref.N). For
    # diffusion at stride=5, this makes ν=1.0 cover 1 MPC window of fine refs
    # at natural pace (= WM training c=1.0 ceiling).
    cfg_ref_stride = cfg.solver.reference.get("stride", None)
    if cfg_ref_stride is None or int(cfg_ref_stride) == 1:
        # Inject diffusion stride into solver config so MPCCSolver.configure
        # picks it up at cost_model build time. OmegaConf assign goes through
        # struct check; use OmegaConf.set_struct False to allow.
        OmegaConf.set_struct(cfg, False)
        cfg.solver.reference.stride = int(diff.stride)
        OmegaConf.set_struct(cfg, True)
        print(f"[diff-MPCC eval] auto-set solver.reference.stride={diff.stride} "
              f"(matches diff.stride). Override with +solver.reference.stride=N if needed.")

    # -- Phase 3: solver + DiffusionMPCCPolicy. gt_refs=False — caller (us)
    # injects dummy refs immediately so cost_model.__init__ has shape, then
    # refreshes on first replan via the policy.
    n_eval = eval_world.n_eval
    D = diff.obs_dim

    # Optional ref-dump dir.
    _run_name = cfg.output.get("run_name") or "eval_diffusion_mpcc"
    _default_dump = (Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
                     / _run_name / "ref_dumps")
    dump_dir_cfg = diff_cfg.get("dump_dir", "default") if diff_cfg else "default"
    if dump_dir_cfg == "default":
        dump_dir = str(_default_dump)
    elif dump_dir_cfg in (None, "null", "", False):
        dump_dir = None
    else:
        dump_dir = str(dump_dir_cfg)

    n_ensemble = int(cfg.solver.get("n_ensemble", 1))

    duplicate_current_history = bool(diff_cfg.get("duplicate_current_history", False)) if diff_cfg else False
    if duplicate_current_history:
        print("[diff-MPCC eval] duplicate_current_history=ON — cond will be K copies of current obs (no past)")

    # Pre-populate solver._references with dummy ReferenceTrajectory objects
    # so cost_model.__init__ has shape. First replan via DiffusionMPCCPolicy.
    # get_action → solver.refresh_reference_trajectories(...) overwrites with
    # real diffusion refs. Each row maps to one (env, candidate) pair: row
    # i*N + e → env i, candidate e (matches EnsembleMPCCSolver's
    # repeat_interleave pattern).
    n_total_refs = n_eval * n_ensemble
    dummy_emb = torch.zeros(2, D)   # min N=2 for arc-length
    dummy_ref = ReferenceTrajectory.from_latents(dummy_emb)

    model, solver, policy, plan_config = _setup_eval_solver_mpcc(
        cfg, eval_world,
        gt_refs=False,
        policy_cls=_policy_cls,                # ← auto-dispatched above
        policy_kwargs={"diff": diff, "action_block": plan_action_block,
                       "dump_dir": dump_dir, "use_first_n": use_first_n,
                       "n_ensemble": n_ensemble,
                       "duplicate_current_history": duplicate_current_history,
                       "view_dims": cfg.solver.reference.get("view_dims", None),
                       "view_weights": cfg.solver.reference.get("view_weights", None),
                       "interpolation": cfg.solver.reference.get("interpolation", "linear")},
    )
    if dump_dir is not None:
        print(f"[diff-MPCC eval] ref dumps → {dump_dir}")
    if n_ensemble > 1:
        print(f"[diff-MPCC eval] ensemble enabled: n_ensemble={n_ensemble} "
              f"(effective batched CEM size = {n_eval} * {n_ensemble} = "
              f"{n_eval * n_ensemble})")

    # Inject dummy refs sized n_eval * n_ensemble. cost_model.__init__ builds
    # nu_scales from these. First replan overwrites.
    solver._references = [
        ReferenceTrajectory.from_latents(dummy_emb.clone())
        for _ in range(n_total_refs)
    ]
    solver._env_ref_indices = np.arange(n_total_refs)

    # -- Phase 4: optional decoder for debug video --------------------------
    debug_cfg = cfg.get("debug", {})
    decoder_ckpt = debug_cfg.get("decoder_checkpoint") if debug_cfg else None
    if decoder_ckpt and cfg.eval.get("debug_video", False):
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("root_utils", _THIS / "utils.py")
        _root_utils = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_root_utils)
        attach_decoder_from_checkpoint = _root_utils.attach_decoder_from_checkpoint
        model = attach_decoder_from_checkpoint(model, decoder_ckpt, map_location="cuda")
        print(f"[diff-MPCC eval] Decoder loaded from {decoder_ckpt}")

    # -- Phase 5: bind policy to world (triggers solver.configure) ----------
    eval_world.world.set_policy(policy)
    _enabled_views = getattr(model, "enabled_views", ("agent", "wrist"))
    _proprio_keys = tuple(getattr(model, "proprio_keys", ()) or ())
    if "wrist" in _enabled_views:
        _attach_wrist_renderers(eval_world.world, img_size=cfg.world.get("width", 224))
        print(f"[renderer] wrist renderer attached (enabled_views={list(_enabled_views)})")
    if "proprio" in _enabled_views:
        _attach_proprio_alias(eval_world.world, _proprio_keys)

    # -- Phase 6: CEM env-space scaler (mirror eval_mpcc_multiview behavior) -
    inner = getattr(solver, "_inner_solver", None)
    if (inner is not None
            and hasattr(inner, "set_action_scaler")
            and hasattr(eval_world.world.policy, "process")
            and "action" in eval_world.world.policy.process):
        inner.set_action_scaler(eval_world.world.policy.process["action"])
        rng = getattr(inner, "env_clip_range", None)
        mode = getattr(inner, "projection_mode", "state")
        if rng is not None and mode != "none":
            print(f"[cem] env-space projection ENABLED  mode={mode}  clip={tuple(rng)}  (phys slot only)")
        else:
            print(f"[cem] env-space projection DISABLED (env_clip_range={rng}, mode={mode})")
        nu_clamp = getattr(inner, "nu_clamp", None)
        nu_init = getattr(inner, "nu_init", None)
        nu_var = getattr(inner, "nu_var_scale", None)
        print(f"[cem] ν shaping: clamp={nu_clamp}, init={nu_init}, var_scale={nu_var}")

    # -- Phase 7: output path ----------------------------------------------
    policy_dir = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
    run_name = cfg.output.get("run_name") or "eval_diffusion_mpcc"
    results_path = policy_dir / run_name
    results_path.mkdir(parents=True, exist_ok=True)
    print(f"[diff-MPCC eval] episodes={np.unique(eval_world.eval_episodes).tolist()}, "
          f"goal_offset={eval_world.goal_offset}, eval_budget={eval_world.eval_budget}, "
          f"n={eval_world.n_eval}, ckpt={Path(diff_cfg.ckpt_path).name}")
    print(f"[diff-MPCC eval] reference.stride={cfg.solver.reference.stride}, "
          f"reference.c={cfg.solver.reference.get('c', 0.3)}")

    # -- Phase 8: run eval (reuse evaluate_tracking_from_dataset) ----------
    callables = (OmegaConf.to_container(cfg.eval.callables, resolve=True)
                 if cfg.eval.get("callables") else None)

    # Build RolloutLogger if requested (+eval.rollout_log=true). Same shape
    # as eval_mpcc_multiview, with diffusion ckpt path baked into meta.
    rollout_logger = None
    if cfg.eval.get("rollout_log", False):
        from latent_modules.eval import RolloutLogger
        method_name = cfg.eval.get("rollout_method_name", "diff-mpcc")
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
            action_block=plan_action_block,
            horizon=int(cfg.plan_config.horizon),
            receding_horizon=int(cfg.plan_config.receding_horizon),
            solver=cfg.solver,
            plan_config=cfg.plan_config,
            world=cfg.world,
            diffusion_ckpt=str(diff_cfg.ckpt_path),
            diffusion_arch={"H": int(diff.H), "K": int(diff.K),
                             "stride": int(diff.stride), "D": int(diff.obs_dim)},
            diffusion_sample={"steps": int(diff.sample_steps),
                              "solver": str(diff.solver),
                              "w_cfg": float(diff.w_cfg)},
            use_first_n=use_first_n,
            n_ensemble=n_ensemble,
            start_offset=cfg.eval.get("start_offset", None),
            reference_stride=int(cfg.solver.reference.stride),
            reference_c=float(cfg.solver.reference.get("c", 0.3)),
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
        debug=bool(cfg.eval.get("debug_video", False)),
        model=model,
        logger=rollout_logger,
    )
    elapsed = time.time() - t0
    print(f"[diff-MPCC eval] done in {elapsed:.1f}s  "
          f"replans={policy._n_replans} ring_updates={policy._n_ring_updates}")

    # -- Phase 9: save results --------------------------------------------
    behavioral = metrics.get("behavioral_aggregate", {})
    success_rate = metrics.get("success_rate", float("nan"))
    print(f"[diff-MPCC eval] success_rate={success_rate:.1f}%  behavioral={behavioral}")

    # Per-env first_success_step (TTF) from evaluate_tracking_from_dataset.
    # Used by analyze/compare scripts to compute mean/median/p10/p90 TTF.
    fs = metrics.get("first_success_step")
    es = metrics.get("episode_successes")
    out_npz = results_path / "diffusion_mpcc_results.npz"
    np.savez(
        out_npz,
        success_rate=success_rate,
        n_replans=policy._n_replans,
        n_ring_updates=policy._n_ring_updates,
        elapsed=elapsed,
        ckpt_path=str(diff_cfg.ckpt_path),
        first_success_step=(np.asarray(fs) if fs is not None
                            else np.zeros(0, dtype=np.int64)),
        episode_successes=(np.asarray(es) if es is not None
                           else np.zeros(0, dtype=bool)),
    )
    out_txt = results_path / "diffusion_mpcc_results.txt"
    out_txt.write_text(
        f"ckpt           : {diff_cfg.ckpt_path}\n"
        f"meta arch      : H={diff.H} K={diff.K} S={diff.stride} D={diff.obs_dim}\n"
        f"sample         : {diff.sample_steps}/{diff.solver}  "
        f"w_cfg={diff.w_cfg}  x_clip={'OFF' if diff.x_clip_abs is None else diff.x_clip_abs}\n"
        f"ref.stride     : {cfg.solver.reference.stride}\n"
        f"ref.c          : {cfg.solver.reference.get('c', 0.3)}\n"
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
    print(f"[diff-MPCC eval] saved {out_npz}  +  {out_txt}")


if __name__ == "__main__":
    run()
