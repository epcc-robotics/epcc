"""Multi-view MPCC evaluation: speed-decoupled trajectory tracking in latent
space using BOTH agentview pixels and wrist_pixels.

Same plumbing as eval_mpc_multiview.py (wrist renderer, proprio alias, past-
action zero pre-fill, scaler injection) but uses MPCC cost model — augmented
action ``(u_phys; ν)`` with ν as virtual velocity controlling progress along
the reference path. ν semantics: ν=0.3=expert pace, ν=1.0=natural pace
(WM-trained ceiling). CEM samples ν alongside physical action; cost.py maps
ν → arc-length advance via ``nu_scale = action_block / (c · ref.N)``.

Plumbing (`_attach_wrist_renderers`, `MultiviewWorldModelPolicy`,
`_setup_eval_world`, `_compute_episode_metrics`,
`evaluate_tracking_from_dataset`) is imported verbatim from
eval_mpc_multiview.py. Only the solver class and the eval entry differ.

Usage:
    python eval_mpcc_multiview.py --config-name multiview_cube_mpcc \
        solver=mpcc_multiview policy=pv_aw_cube_scratch_wm_training/lewm_pv_aw_cube_scratch_epoch_11 \
        eval.dataset_name="Multiview WM/cube_multiview_WM"
"""

import os
import sys

os.environ["MUJOCO_GL"] = "egl"


def _silence_egl_unraisable(unraisable):
    """Drop harmless EGL/OpenGL destructor spam at Python interpreter shutdown."""
    exc_type = unraisable.exc_type
    exc_module = getattr(exc_type, "__module__", "") or ""
    if "OpenGL" in exc_module or "EGL" in exc_type.__name__:
        return
    sys.__unraisablehook__(unraisable)


sys.unraisablehook = _silence_egl_unraisable

import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

# Plumbing reused verbatim from MV MPC entry. Importing the module also runs
# the `_patch_swm_policy_for_wrist_pixels()` monkey-patch at import time — fine
# since the patch is idempotent against double application of the same wrap.
from eval_mpc_multiview import (
    _attach_proprio_alias,
    _attach_wrist_renderers,
    MultiviewWorldModelPolicy,
    _setup_eval_world,
    evaluate_tracking_from_dataset,
)

from latent_modules.mpcc import MPCCSolver


# ---------------------------------------------------------------------------
# Solver (thin MV wrapper)
# ---------------------------------------------------------------------------

class MultiviewMPCCSolver(MPCCSolver):
    """Thin MV wrapper for Hydra ``_target_`` symmetry with MV MPC.

    The base ``MPCCSolver`` works for MV out-of-the-box because:
      - ``ReferenceTrajectory.from_demonstration`` is view-aware (reads
        ``model.enabled_views`` / ``model.proprio_keys``) — see
        latent_modules.mpcc/reference.py.
      - ``MPCCSolver.solve`` pre-encodes pixels + wrist_pixels + proprio in
        a single pass for the ``use_precomputed_emb`` fast path.

    This subclass exists as a Hydra ``_target_`` placeholder so the MV solver
    yaml is parallel to ``MultiviewMPCSolver`` and future MV-specific overrides
    have a clean home.
    """
    pass


# ---------------------------------------------------------------------------
# Reference-span trim (postpad → to-success)   [ported from le-wm local]
# ---------------------------------------------------------------------------

def _compute_trimmed_ref_offsets(cfg, eval_world, margin: int) -> np.ndarray:
    """Per-episode reference offsets cut at the success frame (+ settle margin).

    For postpadded datasets the full episode includes a long freeze/hold tail
    after the goal is reached. Counting it inflates the reference frame count N
    and dilutes ``nu_scale = action_block/(c·stride·N)``. This returns
    ``go_ref`` such that ``set_reference_episodes`` builds the reference over
    ``[start, start+go_ref+1) = [start, cut)`` where
    ``cut = min(start + first_success + 1 + margin, ep_len)``.

    Falls back to the full ``goal_offsets[i]`` for any episode lacking a success
    frame.
    """
    cache_dir_ = cfg.get("cache_dir") or swm.data.utils.get_cache_dir()
    ds = swm.data.HDF5Dataset(
        cfg.eval.dataset_name, keys_to_cache=["success"], cache_dir=cache_dir_
    )
    eps = eval_world.eval_episodes
    starts = eval_world.eval_start_idx
    base_offsets = np.asarray(eval_world.goal_offsets, dtype=np.int64)
    go_ref = base_offsets.copy()
    for i, ep_idx in enumerate(eps):
        st = int(starts[i])
        L = int(ds.lengths[int(ep_idx)])
        ep_data = ds.load_chunk(
            np.array([int(ep_idx)]), np.array([0]), np.array([L])
        )[0]
        succ = ep_data["success"]
        succ = succ.cpu().numpy() if hasattr(succ, "cpu") else np.asarray(succ)
        succ = succ.astype(bool).reshape(-1)
        seg = succ[st:]
        if not seg.any():
            print(f"[MPCC ref-trim][WARN] env {i} ep={int(ep_idx)}: no success "
                  f"frame → full offset {int(base_offsets[i])}")
            continue
        first_success = int(np.argmax(seg))  # relative to st
        cut = min(st + first_success + 1 + margin, L)
        go_ref[i] = cut - st - 1
    return go_ref


# ---------------------------------------------------------------------------
# MPCC solver/policy setup
# ---------------------------------------------------------------------------

def _setup_eval_solver_mpcc(
    cfg,
    eval_world,
    *,
    gt_refs: bool = True,
    policy_cls=None,
    policy_kwargs: dict | None = None,
):
    """Build MPCC solver + policy. Mirrors ``_setup_eval_solver`` in MV MPC.

    Shared between ``eval_mpcc_multiview.run()`` (gt_refs=True, default
    MultiviewWorldModelPolicy) and ``eval_diffusion_mpcc.run()`` (gt_refs=False,
    DiffusionMPCCPolicy).

    Returns ``(model, solver, policy, plan_config)`` or
    ``(None, None, RandomPolicy, None)`` for random policy.

    Args:
        cfg: Hydra config (must have cfg.policy, cfg.solver, cfg.plan_config).
        eval_world: SimpleNamespace from ``_setup_eval_world``.
        gt_refs: If True (default), load dataset GT references via
            ``solver.set_reference_episodes`` + handle ext_init.
            If False, caller must populate ``solver._references`` and
            ``solver._env_ref_indices`` BEFORE ``world.set_policy(policy)``.
        policy_cls: Policy class. Default ``MultiviewWorldModelPolicy``.
        policy_kwargs: Extra kwargs merged with the standard
            ``{solver, config, process, transform}`` policy constructor args.
    """
    if cfg.get("policy", "random") == "random":
        return None, None, swm.policy.RandomPolicy(), None

    model = swm.policy.AutoCostModel(cfg.policy)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    # HF ViT compat patch (transformers 4.46.x): ViTSelfAttention.dropout submodule
    # missing + ViTConfig default attributes privatised. Standard fix applied
    # by cache_latents_perview_aw.py / validate_diffusion_planner.py /
    # train_decoder.py / visualize_diffusion_decoder.py / cache_latents.py /
    # diag_encode_parity_aw.py. Centralised here so all callers
    # (eval_mpcc_multiview, eval_diffusion_mpcc, eval_diffusion_invdyn) inherit.
    import transformers as _hf_transformers
    for _m in model.modules():
        if hasattr(_m, "config") and isinstance(_m.config, _hf_transformers.PretrainedConfig):
            _fresh = type(_m.config)()
            for _k, _v in _fresh.__dict__.items():
                if _k not in _m.config.__dict__:
                    setattr(_m.config, _k, _v)
        if "ViTSelfAttention" in type(_m).__name__ and not hasattr(_m, "dropout"):
            _m.dropout = torch.nn.Dropout(getattr(_m, "dropout_prob", 0.0))

    plan_config = swm.PlanConfig(**OmegaConf.to_container(cfg.plan_config, resolve=True))
    solver = hydra.utils.instantiate(
        cfg.solver, model=model, transform=eval_world.transform["pixels"]
    )
    # Per-step policy.get_action applies same scalers via WorldModelPolicy.
    # Solver hands proprio scalers to ReferenceTrajectory.from_demonstration
    # for AP/AWP variants (no-op for SV/AW where proprio_keys is empty).
    solver._scalers = {k: v for k, v in eval_world.process.items() if k != "action"}

    if gt_refs:
        # Reference span used for nu_scale (= action_block/(c·stride·N)). By
        # default N = full padded episode length. With trim_ref_to_success the
        # reference is cut to the success frame (+settle margin) so the freeze
        # tail no longer inflates N. Rollout budget/scoring are unaffected — we
        # only swap the offsets handed to set_reference_episodes.
        ref_goal_offsets = eval_world.goal_offsets
        _seg = getattr(eval_world, "segment", None)
        if cfg.eval.get("trim_ref_to_success", False) and _seg:
            # Segment mode already pins the reference end at the frame the
            # segment's LAST stage completes. The success-based trim would
            # instead cut at the FIRST dataset success frame after start —
            # which for a multi-stage segment (BC, ABC, ...) is the end of its
            # first stage, silently starving the tracker of any reference for
            # the remaining stages. Keep the segment window and only add the
            # settle margin, which is the part of the trim that helps
            # (a frozen tail past the goal gives the tracker something to
            # converge onto).
            margin = int(cfg.eval.get("trim_settle_margin", 10))
            base = np.asarray(eval_world.goal_offsets, dtype=np.int64)
            lens = np.array(
                [int(eval_world.dataset.lengths[int(e)])
                 for e in eval_world.eval_episodes],
                dtype=np.int64,
            )
            starts = np.asarray(eval_world.eval_start_idx, dtype=np.int64)
            ref_goal_offsets = np.minimum(base + margin, lens - starts - 1)
            print(f"[MPCC ref-trim] segment={_seg}: success-trim SUPPRESSED; "
                  f"using segment window + settle margin {margin} → ref offsets "
                  f"min={int(ref_goal_offsets.min())} "
                  f"median={int(np.median(ref_goal_offsets))} "
                  f"max={int(ref_goal_offsets.max())} "
                  f"(segment window max={int(base.max())})")
        elif cfg.eval.get("trim_ref_to_success", False):
            margin = int(cfg.eval.get("trim_settle_margin", 10))
            ref_goal_offsets = _compute_trimmed_ref_offsets(cfg, eval_world, margin)
            base = np.asarray(eval_world.goal_offsets)
            print(f"[MPCC ref-trim] success+{margin} margin → ref offsets "
                  f"min={int(np.min(ref_goal_offsets))} "
                  f"median={int(np.median(ref_goal_offsets))} "
                  f"max={int(np.max(ref_goal_offsets))} "
                  f"(full would be max={int(np.max(base))})")
        solver.set_reference_episodes(
            eval_world.eval_episodes, eval_world.eval_start_idx, ref_goal_offsets
        )

        # External GT-init from npz (e.g., from a MPC ref_stride run).
        # Path = ``+eval.init_action_npz=<path>``. Optional ν-slot fill value
        # = ``+eval.init_action_nu=<float>`` (default 1.0 = natural pace).
        # Accepts both phys-only (S, n_envs, H, A_phys) and augmented
        # (S, n_envs, H, A_phys+1); ν is added when phys-only.
        ext_init_path = cfg.eval.get("init_action_npz", None)
        if ext_init_path:
            import numpy as _np
            data = _np.load(ext_init_path, allow_pickle=False)
            ext_actions = torch.from_numpy(data["actions"]).float()
            assert ext_actions.dim() == 4
            assert ext_actions.shape[1] == len(eval_world.eval_episodes)
            solver._gt_init_actions = ext_actions
            solver._gt_nu_value = float(cfg.eval.get("init_action_nu", 1.0))
            print(f"[ext GT-init] loaded {ext_init_path}  "
                  f"shape={tuple(ext_actions.shape)} ν_pad={solver._gt_nu_value}")

    # Build policy via injected class (default = MultiviewWorldModelPolicy).
    policy_cls_final = policy_cls or MultiviewWorldModelPolicy
    base_kwargs = {
        "solver": solver,
        "config": plan_config,
        "process": eval_world.process,
        "transform": eval_world.transform,
    }
    if policy_kwargs:
        base_kwargs.update(policy_kwargs)
    policy = policy_cls_final(**base_kwargs)
    return model, solver, policy, plan_config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./config/eval", config_name="multiview_cube_mpcc")
def run(cfg: DictConfig):
    """Run multi-view MPCC evaluation."""

    # Phase 1: env + dataset + transforms (shared with MV MPC).
    eval_world = _setup_eval_world(cfg)
    dataset = eval_world.dataset
    world = eval_world.world
    eval_episodes = eval_world.eval_episodes
    eval_start_idx = eval_world.eval_start_idx
    goal_offsets = eval_world.goal_offsets
    eval_budget = eval_world.eval_budget
    goal_offset = eval_world.goal_offset
    n_eval = eval_world.n_eval

    # Phase 2: model + solver + policy.
    policy_name = cfg.get("policy", "random")
    model, solver, policy, _plan_config = _setup_eval_solver_mpcc(cfg, eval_world)

    # Output path.
    policy_dir = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if policy_name != "random"
        else Path(__file__).parent
    )
    run_name = cfg.output.get("run_name") or "eval_mpcc"
    results_path = policy_dir / run_name
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"[MPCC eval] episodes={np.unique(eval_episodes).tolist()}, "
          f"goal_offset={goal_offset}, eval_budget={eval_budget}, n={n_eval}")

    # Attach view-aware runtime hooks based on model's enabled_views.
    world.set_policy(policy)
    _enabled_views = (
        getattr(model, "enabled_views", ("agent", "wrist"))
        if model is not None else ("agent",)
    )
    _proprio_keys = tuple(getattr(model, "proprio_keys", ()) or ())
    if "wrist" in _enabled_views:
        _attach_wrist_renderers(world, img_size=cfg.world.get("width", 224))
        print(f"[renderer] wrist renderer attached (enabled_views={list(_enabled_views)})")
    else:
        print(f"[renderer] wrist renderer SKIPPED (enabled_views={list(_enabled_views)})")
    if "proprio" in _enabled_views:
        _attach_proprio_alias(world, _proprio_keys)

    # Inject training scaler into CEM (phys slot only — MPCC's physical_dim
    # gates the tile, leaving ν unaffected).
    inner = getattr(solver, "_inner_solver", None) if solver is not None else None
    if (inner is not None
            and hasattr(inner, "set_action_scaler")
            and hasattr(world.policy, "process")
            and "action" in world.policy.process):
        inner.set_action_scaler(world.policy.process["action"])
        rng = getattr(inner, "env_clip_range", None)
        mode = getattr(inner, "projection_mode", "state")
        if rng is not None and mode != "none":
            print(f"[cem] env-space projection ENABLED  mode={mode}  "
                  f"clip={tuple(rng)}  (phys slot only)")
        else:
            print(f"[cem] env-space projection DISABLED "
                  f"(env_clip_range={rng}, mode={mode})")
        # ν shaping config trace.
        nu_clamp = getattr(inner, "nu_clamp", None)
        nu_init = getattr(inner, "nu_init", None)
        nu_var = getattr(inner, "nu_var_scale", None)
        print(f"[cem] ν shaping: clamp={nu_clamp}, init={nu_init}, var_scale={nu_var}")
    else:
        print("[cem] scaler not attached (no inner solver / no policy.process['action'])")

    # Build RolloutLogger if requested (opt-in via +eval.rollout_log=true).
    # Single shared writer for all controllers — see latent_modules.eval/.
    rollout_logger = None
    if cfg.eval.get("rollout_log", False):
        from latent_modules.eval import RolloutLogger
        method_name = cfg.eval.get("rollout_method_name", "gt-mpcc")
        task_name = cfg.eval.get("rollout_task_name", "cube-mv")
        rollout_logger = RolloutLogger(
            method=method_name,
            task=task_name,
            out_dir=results_path,
            n_envs=n_eval,
            eval_budget=eval_budget,
        )
        rollout_logger.set_episodes(
            episode_idx=eval_episodes.tolist(),
            start_step=eval_start_idx.tolist(),
            goal_offset=(goal_offsets.tolist() if hasattr(goal_offsets, "tolist")
                         else [int(goal_offsets)] * n_eval),
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
            solver=cfg.solver,
            plan_config=cfg.plan_config,
            world=cfg.world,
        )
        rollout_logger.attach_demo_ttc(cfg.eval.get("demo_ttc_path", None))
        print(f"[rollout_log] enabled — writing to {results_path}")

    # Run evaluation. Reuse MV MPC's eval loop verbatim — MPCC's solver has
    # no-op ``advance_steps``/``reset_steps`` so the calls are harmless.
    start_time = time.time()
    metrics = evaluate_tracking_from_dataset(
        world,
        dataset,
        solver=solver,
        episodes_idx=eval_episodes.tolist(),
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=goal_offsets if goal_offsets is not None else goal_offset,
        eval_budget=eval_budget,
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
        save_video=cfg.eval.get("save_video", True),
        debug=cfg.eval.get("debug_video", False),
        model=model if cfg.eval.get("debug_video", False) else None,
        logger=rollout_logger,
    )
    elapsed = time.time() - start_time

    # Summary.
    sr = metrics["success_rate"]
    print(f"\n[MPCC eval] Success rate: {sr:.1f}%")
    print(f"[MPCC eval] Eval time: {elapsed:.1f}s")

    # Save results.
    results_file = results_path / (cfg.output.get("filename") or "mpcc_results.txt")
    with open(results_file, "w") as f:
        f.write(f"success_rate: {sr:.1f}%\n")
        f.write(f"eval_budget: {eval_budget}\n")
        f.write(f"goal_offset: {goal_offset}\n")
        f.write(f"n_eval: {n_eval}\n")
        f.write(f"eval_time: {elapsed:.1f}s\n")

        bag = metrics.get("behavioral_aggregate")
        if bag is not None:
            n_e = bag["n_envs"]
            f.write("\n=== Behavioral aggregate ===\n")
            f.write(f"grasped:      {bag['grasped']}/{n_e} = {100*bag['grasped']/n_e:.1f}%\n")
            f.write(f"no_drop:      {bag['no_drop']}/{n_e} = {100*bag['no_drop']/n_e:.1f}%\n")
            f.write(f"reached_goal: {bag['reached_goal']}/{n_e} = {100*bag['reached_goal']/n_e:.1f}%\n")
            f.write(f"full_GTR:     {bag['full']}/{n_e} = {100*bag['full']/n_e:.1f}%\n")
            f.write(f"mean_final_dist: {bag['mean_final_dist']:.4f}m\n")

        f.write("\n=== Per-env ===\n")
        beh = metrics.get("behavioral") or [{}] * len(metrics["episode_successes"])
        for i in range(len(metrics["episode_successes"])):
            ep_idx = int(eval_episodes[i]) if i < len(eval_episodes) else -1
            m = beh[i] if i < len(beh) else {}
            flags = (
                ("G" if m.get("grasped") else "-")
                + ("T" if m.get("no_drop") else "-")
                + ("R" if m.get("reached_goal") else "-")
            )
            freeze_step = int(m.get("freeze_step", -1))
            f.write(
                f"env_{i}: ep={ep_idx}, success={bool(metrics['episode_successes'][i])}, "
                f"first_success={int(metrics['first_success_step'][i])}, "
                f"flags=[{flags}], "
                f"final_dist={m.get('final_dist', float('nan')):.4f}m, "
                f"grasp_run={m.get('longest_grasp_run', 0)}step, "
                f"first_grasp={m.get('first_grasp_step', -1)}, "
                f"drop_step={m.get('drop_step', -1)}, "
                f"freeze_step={freeze_step}\n"
            )
    print(f"[MPCC eval] Results saved to {results_file}")

    # ---- Per-solve npz dump (cfg.eval.save_data) ----
    if cfg.eval.get("save_data", False) and solver is not None and getattr(solver, "_solve_log", None):
        npz_file = results_path / "solve_log.npz"
        log = solver._solve_log
        # Augmented action (phys + ν): (S, n_envs, H, A_phys+1)
        actions = torch.stack([entry["actions"] for entry in log], dim=0)
        def _stack(field):
            return np.array([entry.get(field, []) for entry in log], dtype=np.float32)
        np.savez(
            npz_file,
            actions=actions.cpu().numpy(),
            costs=_stack("costs"),
            init_l2=_stack("init_l2"),
            final_l2=_stack("final_l2"),
            init_aabs=_stack("init_aabs"),
            final_aabs=_stack("final_aabs"),
            init_nu=_stack("init_nu"),
            final_nu=_stack("final_nu"),
            n_solves=len(log),
            n_envs=len(eval_episodes),
            physical_dim=int(getattr(solver, "_physical_action_dim", actions.shape[-1] - 1)),
            episodes=np.array(eval_episodes, dtype=np.int64),
        )
        print(f"[MPCC eval] solve_log.npz saved to {npz_file}  "
              f"(S={len(log)}, n_envs={len(eval_episodes)}, A_aug={actions.shape[-1]})")


if __name__ == "__main__":
    run()
