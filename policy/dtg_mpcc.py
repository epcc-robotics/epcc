"""Task-orthogonal DTG+MPCC — diffusion trajectory generator + MPCC tracking.

Wraps the legacy ``DiffusionMPCCPolicy`` from ``eval_diffusion_mpcc.py``.
Mirrors the run() setup in eval_diffusion_mpcc.py:313 (Phases 2-6):
  * auto-set ``solver.reference.stride`` to ``diff.stride``
  * build diffusion planner with use_first_n validation
  * call ``_setup_eval_solver_mpcc(gt_refs=False)`` with FULL policy_kwargs
  * pre-populate ``solver._references`` with dummy ReferenceTrajectory shells
    sized n_eval × n_ensemble (cost_model.__init__ needs shape; first replan
    via the inner policy overwrites with real diffusion refs)
  * inject CEM env-space ``action_scaler`` into the inner solver

Without these pieces the MPCC solver runs with garbage references and
unscaled CEM samples → policy never grasps. Validated at 70% in v1 with
batch_size=20; v2 must reproduce that anchor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from policy.base import SolverPolicy


class DTGMPCCPolicy(SolverPolicy):
    """Diffusion trajectory generator + MPCC inner-loop tracking."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._inner = None
        self._solver = None
        self._n_replans = 0

    def reset(self, world: Any, cfg: DictConfig | None = None, eval_world: Any | None = None) -> None:
        if cfg is None or eval_world is None:
            raise ValueError("DTGMPCCPolicy.reset requires cfg + eval_world")

        from omegaconf import OmegaConf
        from eval_mpcc_multiview import _setup_eval_solver_mpcc
        from latent_diffusion_cd.planner import (
            is_cleandiffuser_ckpt, build_cleandiffuser_diff_planner,
        )
        from latent_modules.mpcc.reference import ReferenceTrajectory

        diff_cfg = cfg.get("diffusion", None)
        if diff_cfg is None or not diff_cfg.get("ckpt_path"):
            raise ValueError("DTGMPCCPolicy.reset requires cfg.diffusion.ckpt_path")
        # Fail here, not minutes later inside torch.load: a nonexistent path (a placeholder,
        # or a typo that looks real) would otherwise make is_cleandiffuser_ckpt() return
        # False and silently route to the legacy planner branch.
        from pathlib import Path as _Path
        if not _Path(str(diff_cfg.ckpt_path)).is_file():
            raise FileNotFoundError(f"cfg.diffusion.ckpt_path does not exist: {diff_cfg.ckpt_path}")

        # Phase 2 — build planner with auto-dispatch on ckpt format.
        if is_cleandiffuser_ckpt(str(diff_cfg.ckpt_path)):
            from eval_diffusion_mpcc import CleanDiffuserDiffMPCCPolicy as DiffusionMPCCPolicy
            diff = build_cleandiffuser_diff_planner(
                ckpt_path=str(diff_cfg.ckpt_path), device="cuda",
            )
        else:
            from eval_diffusion_mpcc import DiffusionMPCCPolicy
            from latent_diffusion.planner import build_diffusion_planner
            overrides = (OmegaConf.to_container(diff_cfg.overrides, resolve=True)
                         if diff_cfg.get("overrides") else None)
            diff = build_diffusion_planner(
                diff_cfg.ckpt_path, device="cuda", overrides=overrides,
            )

        plan_action_block = int(cfg.plan_config.action_block)
        use_first_n = diff_cfg.get("use_first_n", None)
        effective_H = int(use_first_n) if use_first_n is not None else diff.H
        if effective_H < 2:
            raise ValueError(
                f"effective diffusion H ({effective_H}) < 2 — MPCC ReferenceTrajectory "
                "needs ≥2 waypoints for arc-length parametrization."
            )

        # Auto-set solver.reference.stride to diff.stride (legacy line 371-380).
        cfg_ref_stride = cfg.solver.reference.get("stride", None)
        if cfg_ref_stride is None or int(cfg_ref_stride) == 1:
            OmegaConf.set_struct(cfg, False)
            cfg.solver.reference.stride = int(diff.stride)
            OmegaConf.set_struct(cfg, True)
        # Greppable record of the effective stride (also lands in meta.json, which dumps this
        # cfg object). A yaml stride other than None/1 is NOT synced — it is printed here so a
        # mismatch with diff.stride is visible in the job log.
        print(f"[dtg-mpcc] reference.stride={int(cfg.solver.reference.stride)} "
              f"diff.stride={int(diff.stride)} diff.K={int(diff.K)} diff.H={int(diff.H)} "
              f"action_block={plan_action_block}")

        # Phase 3 — assemble policy_kwargs MATCHING legacy line 420-426.
        n_eval = eval_world.n_eval
        n_ensemble = int(cfg.solver.get("n_ensemble", 1))
        duplicate_current_history = bool(diff_cfg.get("duplicate_current_history", False))

        dump_dir_cfg = diff_cfg.get("dump_dir", None)
        dump_dir = None if dump_dir_cfg in (None, "null", "", False) else str(dump_dir_cfg)

        policy_kwargs = {
            "diff": diff,
            "action_block": plan_action_block,
            "dump_dir": dump_dir,
            "use_first_n": use_first_n,
            "n_ensemble": n_ensemble,
            "duplicate_current_history": duplicate_current_history,
            "view_dims": cfg.solver.reference.get("view_dims", None),
            "view_weights": cfg.solver.reference.get("view_weights", None),
            "interpolation": cfg.solver.reference.get("interpolation", "linear"),
        }

        # Phase 3.5 — cfg.policy clash: legacy AutoCostModel reads cfg.policy as
        # the config-group str. Swap, build, restore.
        legacy_policy_group = str(cfg.task.policy_config_group)
        OmegaConf.set_struct(cfg, False)
        saved_policy = cfg.policy
        cfg.policy = legacy_policy_group
        try:
            _model, _solver, inner_policy, _plan_config = _setup_eval_solver_mpcc(
                cfg, eval_world,
                gt_refs=False,
                policy_cls=DiffusionMPCCPolicy,
                policy_kwargs=policy_kwargs,
            )
        finally:
            cfg.policy = saved_policy
            OmegaConf.set_struct(cfg, True)

        # Phase 4 — pre-populate dummy refs sized n_eval × n_ensemble (legacy
        # line 412-441). cost_model needs shape; first replan overwrites.
        D = diff.obs_dim
        dummy_emb = torch.zeros(2, D)
        n_total_refs = n_eval * n_ensemble
        _solver._references = [
            ReferenceTrajectory.from_latents(dummy_emb.clone())
            for _ in range(n_total_refs)
        ]
        _solver._env_ref_indices = np.arange(n_total_refs)

        self._model = _model
        self._solver = _solver
        self._inner = inner_policy
        self._inner.set_env(world.envs)

        # Phase 5 — inject CEM env-space action_scaler (legacy line 466-471).
        # The inner_policy carries the `process` scaler dict built by
        # _setup_eval_solver_mpcc; access via inner_policy.process["action"].
        inner_cem = getattr(_solver, "_inner_solver", None)
        if (
            inner_cem is not None
            and hasattr(inner_cem, "set_action_scaler")
            and hasattr(inner_policy, "process")
            and inner_policy.process is not None
            and "action" in inner_policy.process
        ):
            inner_cem.set_action_scaler(inner_policy.process["action"])

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        if self._inner is None:
            raise RuntimeError("DTGMPCCPolicy.get_action called before reset(...)")
        # Detect actual replan via inner's _action_buffer state (see MPCCPolicy
        # for rationale). Increments once per receding_horizon × action_block,
        # not per get_action call.
        buf = getattr(self._inner, "_action_buffer", None)
        if buf is None or len(buf) == 0:
            self._n_replans += 1
        return self._inner.get_action(info_dict, **kwargs)

    @property
    def n_replans(self) -> int:
        """Total CEM solves since reset (one per receding_horizon refill)."""
        return int(self._n_replans)
