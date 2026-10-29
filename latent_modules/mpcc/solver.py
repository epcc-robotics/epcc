"""MPCCSolver: Solver protocol adapter for latent MPCC."""

from __future__ import annotations

import os
from typing import Any

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from gymnasium.spaces import Box
from omegaconf import DictConfig
from torch import nn

from latent_modules.mpcc.cost import MPCCCostModel
from latent_modules.mpcc.reference import ReferenceTrajectory


class MPCCSolver:
    """MPCC solver satisfying the stable_worldmodel Solver protocol.

    Augments the physical action space with a virtual velocity ν,
    delegates planning to an inner CEM solver, then strips ν from output.

    ν shaping (clamp / init mean / init var) lives entirely on the inner
    CEM via ``physical_dim``/``nu_clamp``/``nu_init``/``nu_var_scale``
    kwargs. This solver only handles dim augmentation/stripping at the
    boundaries.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_solver: DictConfig,
        cost_weights: DictConfig,
        reference: DictConfig,
        transform: Any = None,
    ):
        self._model = model
        self._device = next(model.parameters()).device
        self._cost_weights = cost_weights
        self._inner_solver_cfg = inner_solver
        self._reference_cfg = reference
        self._transform = transform

        self._references: list[ReferenceTrajectory] = []
        self._env_ref_indices: np.ndarray | None = None
        self._action_block: int = 1
        # Per-env env-step counter for theta_anchor="time" (MPC-like time-indexed
        # θ_0). Advanced once per world.step by the eval loop's advance_steps().
        # None → counter inactive (default project-anchor needs no counter).
        self._env_ref_steps: np.ndarray | None = None

        # Optional proprio scalers — set externally by caller (e.g.,
        # eval entry point) via ``solver._scalers = process_dict``. Used
        # by ``set_reference_episodes`` for AP/AWP variants. SV/AW path
        # leaves this as {} (no proprio in encode).
        self._scalers: dict = {}

        # Per-solve log accumulator (eval.save_data dump). Stores augmented
        # elite-mean action [phys; ν] before ν is stripped, plus init/final
        # cost+l2+|a|+ν per env.
        self._solve_log: list[dict] = []

        # GT-init plumbing — when set externally (eval.init_action_npz):
        #   - shape (S, n_envs, H, A_phys)     → pad ν with ``_gt_nu_value``
        #   - shape (S, n_envs, H, A_phys+1)   → pass through as augmented
        # Each solve() pops one slice. ``_gt_nu_value`` defaults to 1.0
        # (assumes external action came from a fast/natural-pace MPC).
        self._gt_init_actions: torch.Tensor | None = None
        self._gt_solve_count: int = 0
        self._gt_nu_value: float = 1.0

    # -- reference trajectories ---------------------------------------------

    def set_reference_episodes(
        self,
        episode_indices: np.ndarray,
        start_steps: np.ndarray,
        goal_offset: int | np.ndarray,
    ) -> None:
        """Build one reference per unique (episode, start, offset) triple.

        Each reference covers ``[start, start + goal_offset]`` encoded in
        latent space at ``frameskip=ref_cfg.frameskip`` (default 1, fine).
        ``goal_offset`` can be a scalar (same for all) or per-env array.

        View-aware: ``ReferenceTrajectory.from_demonstration`` reads
        ``model.enabled_views``/``model.proprio_keys`` and passes
        ``self._scalers`` for proprio normalisation.

        Must be called before ``configure()``.
        """
        ref_cfg = self._reference_cfg
        goal_offsets = np.array(goal_offset)
        keys = [(int(ep), int(st), int(go))
                for ep, st, go in zip(episode_indices, start_steps, goal_offsets)]
        unique_keys = list(dict.fromkeys(keys))

        print(f"[MPCC] Building {len(unique_keys)} reference trajectories ...")

        key_to_idx: dict[tuple[int, int, int], int] = {}
        references = []
        for ep, st, go in unique_keys:
            ref = ReferenceTrajectory.from_demonstration(
                model=self._model,
                transform=self._transform,
                dataset_name=ref_cfg.dataset_name,
                episode_idx=ep,
                start_step=st,
                end_step=st + go + 1,
                chunk_size=ref_cfg.get("chunk_size", 64),
                cache_dir=ref_cfg.get("cache_dir", None),
                frameskip=ref_cfg.get("frameskip", 1),
                scalers=self._scalers,
                view_dims=ref_cfg.get("view_dims", None),
                view_weights=ref_cfg.get("view_weights", None),
                interpolation=ref_cfg.get("interpolation", "linear"),
            )
            key_to_idx[(ep, st, go)] = len(references)
            references.append(ref)

        self._references = references
        self._env_ref_indices = np.array([key_to_idx[k] for k in keys])

    # -- Solver ----------------------------------------------------

    def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
        self._physical_action_dim = int(np.prod(action_space.shape[1:])) * config.action_block
        self._action_block = config.action_block
        self._n_envs = n_envs

        # theta_anchor="time" needs a per-env env-step counter (advanced each
        # world.step by the eval loop). Allocate only in that mode so the
        # default project-anchor path is untouched (counter stays None →
        # _ref_step absent → cost falls back to projection).
        if str(self._cost_weights.get("theta_anchor", "project")) == "time":
            self._env_ref_steps = np.zeros(n_envs, dtype=np.float64)

        # references must be set
        assert self._references is not None, "References must be set before configuring the solver"

        # build cost model — pass action_block + c so nu_scale = action_block/(c·N)
        # gives ν=1.0 = natural pace per CEM step. c default 0.3 (cube AW expert
        # ratio); override via reference yaml: ``reference.c: <ratio>``.
        c = float(self._reference_cfg.get("c", 0.3))
        # Stride: env-step span per reference frame. Default 1 (dataset refs:
        # 1 frame = 1 env-step). For diffusion-MPCC (frame stride S), set via
        # ``reference.stride: S`` so nu_scale = action_block / (c · stride · ref.N)
        # absorbs the non-unit frame density.
        stride = int(self._reference_cfg.get("stride", 1))
        # Oracle (variant A): swap the WM predictor for real-env dynamics inside
        # the cost rollout. Diagnostic only — see cost_realenv.py.
        _oracle = self._cost_weights.get("oracle_realenv", None)
        _CostCls = MPCCCostModel
        _extra: dict = {}
        if _oracle is not None and bool(_oracle.get("enable", False)):
            from latent_modules.mpcc.cost_realenv import RealEnvMPCCCostModel
            _CostCls = RealEnvMPCCCostModel
            _extra = dict(
                action_mean=tuple(_oracle.get("action_mean", [-0.0022, 0.0036])),
                action_std=tuple(_oracle.get("action_std", [0.1975, 0.1916])),
                env_damping=float(_oracle.get("env_damping", 0.35)),
                phys_per_step=int(np.prod(action_space.shape[1:])),
            )
            print(f"[MPCC] *** ORACLE real-env dynamics ENABLED *** {_extra}", flush=True)
        self.cost_model = _CostCls(
            world_model=self._model,
            references=self._references,
            physical_action_dim=self._physical_action_dim,
            action_block=config.action_block,
            c=c,
            stride=stride,
            q_contour=self._cost_weights.contour,
            q_progress=self._cost_weights.progress,
            q_action=self._cost_weights.action,
            # action_smooth is the generic name (cost.py matches); grip_smooth
            # is kept as a backward-compat fallback so older recipes
            # (which still say grip_smooth)
            # keep working unchanged.
            q_action_smooth=float(
                self._cost_weights.get(
                    "action_smooth", self._cost_weights.get("grip_smooth", 0.0)
                )
            ),
            action_smooth_all_dims=bool(
                self._cost_weights.get(
                    "action_smooth_all_dims",
                    self._cost_weights.get("grip_smooth_all_dims", False),
                )
            ),
            z_norm=self._cost_weights.get("z_norm", False),
            horizon_weight=self._cost_weights.get("horizon_weight", "uniform"),
            contour_mode=self._cost_weights.get("contour_mode", "legacy"),
            q_lag=self._cost_weights.get("lag", 0.0),
            ref_target=self._cost_weights.get("ref_target", "interp"),
            theta_anchor=self._cost_weights.get("theta_anchor", "project"),
            q_rot=float(self._cost_weights.get("rotation", 0.0)),
            q_angvel=float(self._cost_weights.get("angvel", 0.0)),
            rot_oracle_gain=float(self._cost_weights.get("rot_oracle_gain", 0.0)),
            rot_probe_path=str(self._cost_weights.get("rot_probe_path", "")),
            q_agent=float(self._cost_weights.get("agent", 0.0)),
            agent_probe_path=str(self._cost_weights.get("agent_probe_path", "")),
            q_block=float(self._cost_weights.get("block", 0.0)),
            block_probe_path=str(self._cost_weights.get("block_probe_path", "")),
            q_blockvel=float(self._cost_weights.get("block_vel", 0.0)),
            **_extra,
        ).to(self._device)

        # OOD penalties (ported from MPC). ensemble JRD targets the MPCC
        # WM-exploitation failure mode. Disabled by default.
        _dens = self._cost_weights.get("density", None)
        if _dens is not None and _dens.get("enable", False):
            from latent_modules.mpc.density import DensityProxy
            _dp = DensityProxy(_dens.npz, _dens.meta, self._device)
            if _dens.get("q", None) is not None:
                _dp.q = float(_dens.q)
            self.cost_model._density_proxy = _dp
            self.cost_model._density_lambda = float(_dens.get("lambda", 1.0))
            print(f"[MPCC] density proxy ENABLED lambda={self.cost_model._density_lambda} q={_dp.q:.3f}")
        _ens = self._cost_weights.get("ensemble", None)
        if _ens is not None and _ens.get("enable", False):
            from latent_modules.mpc.ensemble_proxy import EnsembleProxy
            _ep = EnsembleProxy(_ens.ckpt, _ens.meta, self._device)
            if _ens.get("eps", None) is not None:
                _ep.eps = float(_ens.eps)
            self.cost_model._ensemble_proxy = _ep
            self.cost_model._ensemble_lambda = float(_ens.get("lambda", 1.0))
            print(f"[MPCC] ensemble proxy ENABLED lambda={self.cost_model._ensemble_lambda} "
                  f"eps={_ep.eps:.3f} K={_ep.ens.ensemble_size}")

        # inner CEM solver with augmented action space (physical + ν)
        aug_dim = self._physical_action_dim + 1
        aug_low = np.full((n_envs, aug_dim), -1.0, dtype=np.float32)
        aug_high = np.full((n_envs, aug_dim), 1.0, dtype=np.float32)
        aug_low[:, -1] = 0.0  # ν ≥ 0 (forward progress only)

        self._inner_solver = hydra.utils.instantiate(self._inner_solver_cfg, model=self.cost_model)
        # Inject env action dim (per env-step) so the inner CEM can locate the
        # gripper-dim inside the augmented action vector. The inner CEM's
        # ``_action_dim`` here is ``aug_dim`` (phys + ν), so its MPC-fallback
        # for phys_per_step is wrong; set it from the original env action space.
        self._inner_solver.phys_per_step = int(np.prod(action_space.shape[1:]))
        self._inner_solver.configure(
            action_space=Box(low=aug_low, high=aug_high),
            n_envs=n_envs,
            config=swm.PlanConfig(
                horizon=config.horizon,
                receding_horizon=config.receding_horizon,
                action_block=1,
            ),
        )

    def reset_steps(self) -> None:
        pass

    def advance_steps(self, n: int = 1) -> None:
        pass

    def refresh_reference_trajectories(self, new_references: list[ReferenceTrajectory]) -> None:
        """Hot-swap per-env ReferenceTrajectory list and rebuild cost_model state.

        Used by diffusion-MPCC's per-replan refresh: every replan the diffusion
        planner emits fresh refs; we propagate them to BOTH the solver's list
        and the cost_model's nn.ModuleList + nu_scales (cost.py reads from
        ``self.references``, not from the solver's list).

        Mirrors the encapsulation pattern of ``latent_modules.mpc.MPCCostModel.
        refresh_references``: solver delegates to cost_model so the cost_model
        owns its own state.

        Args:
            new_references: list[ReferenceTrajectory], one per (env, candidate)
                row. Length must match the n_envs * N_ensemble allocation.
        """
        self._references = list(new_references)
        if hasattr(self, "cost_model") and self.cost_model is not None:
            self.cost_model.refresh_reference_trajectories(new_references)

    @property
    def action_dim(self) -> int:
        return self._physical_action_dim

    @property
    def n_envs(self) -> int:
        return self._inner_solver.n_envs

    def advance_steps(self, n: int = 1):
        """Advance the per-env env-step counter (theta_anchor="time" only).

        Called once per ``world.step`` by the eval loop (polymorphic — the
        default project-anchor leaves ``_env_ref_steps`` None, making this a
        no-op). One env-step → +n on the counter; cost converts to Δθ via
        ``nu_scale/action_block`` so the time-anchored θ_0 tracks natural pace.
        """
        if self._env_ref_steps is not None:
            self._env_ref_steps += n

    def reset_steps(self):
        """Reset the env-step counter to 0 (new episode). No-op when inactive."""
        if self._env_ref_steps is not None:
            self._env_ref_steps[:] = 0.0

    @property
    def horizon(self) -> int:
        return self._inner_solver.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        # GT-init override: pop the next slice if a list is loaded externally.
        if (init_action is None
                and self._gt_init_actions is not None
                and self._gt_solve_count < self._gt_init_actions.shape[0]):
            init_action = self._gt_init_actions[self._gt_solve_count].to(self._device)
            self._gt_solve_count += 1

        # Inject per-env reference index for cost model.
        if self._env_ref_indices is not None:
            info_dict["_ref_idx"] = torch.from_numpy(self._env_ref_indices).long()
        # Inject the env-step counter for theta_anchor="time" (float; cost
        # scales it by Δθ_per_env-step). Absent in default project-anchor mode.
        if self._env_ref_steps is not None:
            info_dict["_ref_step"] = torch.from_numpy(self._env_ref_steps.copy())

        # Pre-encode pixels (and any MV view tensors) ONCE per solve so the
        # encoder doesn't run on every CEM iteration. CEMSolver expands
        # info_dict tensors to (B, S, ...) before calling cost_model, so the
        # cached emb (B, T_hist, D) becomes (B, S, T_hist, D) which
        # JEPA.rollout's use_precomputed_emb branch picks up. Mirrors
        # `latent_modules.mpc.MPCSolver.solve` — 10× speedup at MV cost-eval.
        with torch.no_grad():
            encode_input: dict[str, Any] = {
                "pixels": info_dict["pixels"].to(self._device, non_blocking=True),
            }
            # Pass through optional MV streams so encode populates them too.
            # Absent keys are skipped (SV path is unchanged).
            for _key in ("wrist_pixels", "proprio_gripper_opening",
                          "proprio_effector_pos"):
                if _key in info_dict and torch.is_tensor(info_dict[_key]):
                    encode_input[_key] = info_dict[_key].to(self._device, non_blocking=True)
            self._model.encode(encode_input)
            info_dict["emb"] = encode_input["emb"]   # (B, T_hist, D)

            # [dist-shift ablation] cache current latent (last history frame, one row
            # per real env) each solve → per-replan deployment-latent trace for the
            # WM↔deployment observation-gap (sliced-W1). Read post-loop by
            # eval_mpcc_multiview.run. Tiny: (n_envs, D) per replan.
            self._last_cur_emb = encode_input["emb"][:, -1, :].detach().to("cpu").float()
            if not hasattr(self, "_cur_emb_history"):
                self._cur_emb_history = []
            self._cur_emb_history.append(self._last_cur_emb.numpy().copy())

        # Augment warm-start with ν.
        # - phys-only init (last dim = physical_action_dim): pad with
        #   ``_gt_nu_value`` if GT-init came from external npz, otherwise
        #   ``nu_init`` from inner CEM (cold-start consistency).
        # - already-augmented init (last dim = physical_action_dim+1):
        #   pass through unchanged.
        if init_action is not None:
            A_in = init_action.shape[-1]
            if A_in == self._physical_action_dim:
                # Decide ν fill: GT externally loaded → _gt_nu_value (1.0 by
                # default = natural pace assumption); otherwise nu_init from CEM.
                if self._gt_init_actions is not None and self._gt_solve_count > 0:
                    fill = float(self._gt_nu_value)
                else:
                    fill = float(getattr(self._inner_solver, "nu_init", None) or 0.0)
                nu_pad = torch.full(
                    (*init_action.shape[:-1], 1),
                    fill_value=fill,
                    dtype=init_action.dtype,
                    device=init_action.device,
                )
                init_action = torch.cat([init_action, nu_pad], dim=-1)
            elif A_in != self._physical_action_dim + 1:
                raise ValueError(
                    f"init_action last dim {A_in} ≠ phys ({self._physical_action_dim}) "
                    f"or phys+1 ({self._physical_action_dim + 1})"
                )

        outputs = self._inner_solver.solve(info_dict, init_action=init_action)

        # Per-solve log — capture augmented action (phys+ν) BEFORE ν strip
        # so downstream analysis sees the full plan including ν per chunk.
        self._solve_log.append({
            "actions":    outputs["actions"].clone(),  # (n_envs, H, A_phys+1)
            "costs":      list(outputs.get("costs", [])),
            "init_l2":    list(outputs.get("init_l2", [])),
            "final_l2":   list(outputs.get("final_l2", [])),
            "init_aabs":  list(outputs.get("init_aabs", [])),
            "final_aabs": list(outputs.get("final_aabs", [])),
            "init_nu":    list(outputs.get("init_nu", [])),
            "final_nu":   list(outputs.get("final_nu", [])),
        })

        # Strip ν — WorldModelPolicy only sees physical actions.
        outputs["actions"] = outputs["actions"][..., :self._physical_action_dim]
        return outputs
