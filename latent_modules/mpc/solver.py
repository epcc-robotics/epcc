"""MPCSolver: Time-indexed trajectory tracking MPC using CEM.

Simpler than MPCC — no virtual velocity, no arc-length parametrization.
At each step t, the solver minimizes latent distance to reference[t:t+H].
"""

from __future__ import annotations

from typing import Any

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from latent_modules.mpc.cost import MPCCostModel, MPCCostModelDebug


class MPCSolver:
    """MPC solver satisfying the stable_worldmodel Solver protocol.

    Delegates planning to an inner CEM solver with a time-indexed tracking
    cost. Subclass hook ``COST_MODEL_CLS`` selects which cost model is
    built in :meth:`configure`; :class:`MPCSolverDebug` uses this to plug
    in :class:`MPCCostModelDebug` for per-env breakdown collection.
    """

    COST_MODEL_CLS = MPCCostModel

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

        self._references: list[torch.Tensor] = []  # list of (N, D) embeddings
        self._env_ref_indices: np.ndarray | None = None
        self._env_ref_steps: np.ndarray | None = None  # current step per env
        self._action_block: int = 1
        # Per-solve log accumulator (for eval.save_data dump). Each entry =
        # one solve, dict with elite mean, init/final l2 + cost + |a|, and
        # per-env ref_step at solve time.
        self._solve_log: list[dict] = []
        # [dist-shift ablation] per-solve current emb (cur_emb) — eval dumps these
        # to rollout_latents.npz for the WM↔deployment observation-gap (sliced-W1).
        self._cur_emb_history: list = []

        # GT-init plumbing — when set externally (eval.init_action_npz), each
        # solve() pops one slice (n_envs, H, A_phys) and passes as init_action.
        # Falls back to None (cold-start) when list exhausted or unset.
        self._gt_init_actions: torch.Tensor | None = None  # (S, n_envs, H, A_phys)
        self._gt_solve_count: int = 0

    # -- reference trajectories ------------------------------------------------

    def set_reference_episodes(
        self,
        episode_indices: np.ndarray,
        start_steps: np.ndarray,
        goal_offset: int | np.ndarray,
    ) -> None:
        """Build one reference per unique (episode, start, offset) triple.

        Each reference covers [start, start + goal_offset] encoded in latent
        space. ``goal_offset`` can be a scalar (same for all) or per-env array.
        Must be called before configure().
        """
        ref_cfg = self._reference_cfg
        goal_offsets = np.array(goal_offset)
        keys = [(int(ep), int(st), int(go))
                for ep, st, go in zip(episode_indices, start_steps, goal_offsets)]
        unique_keys = list(dict.fromkeys(keys))

        print(f"[MPC] Building {len(unique_keys)} reference trajectories ...")

        key_to_idx: dict[tuple[int, int, int], int] = {}
        references = []
        for ep, st, go in tqdm(unique_keys):
            embs = self._encode_episode(
                dataset_name=ref_cfg.dataset_name,
                episode_idx=ep,
                start_step=st,
                end_step=st + go + 1,
                chunk_size=ref_cfg.get("chunk_size", 64),
                cache_dir=ref_cfg.get("cache_dir", None),
            )
            key_to_idx[(ep, st, go)] = len(references)
            references.append(embs)

        self._references = references
        self._env_ref_indices = np.array([key_to_idx[k] for k in keys])

    def _encode_episode(
        self,
        dataset_name: str,
        episode_idx: int,
        start_step: int,
        end_step: int,
        chunk_size: int = 64,
        cache_dir: str | None = None,
    ) -> torch.Tensor:
        """Encode demo segment into latent embeddings (N, D). Single-view."""
        kwargs: dict[str, Any] = {"name": dataset_name, "frameskip": 1, "keys_to_load": ["pixels"]}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        episode = swm.data.HDF5Dataset(**kwargs).load_episode(episode_idx)
        pixels = episode["pixels"][start_step:end_step]

        transform = self._transform
        frames = torch.stack([transform(p) for p in pixels])

        is_prejepa = hasattr(self._model, 'extra_encoders') and self._model.extra_encoders
        encode_kwargs = {}
        if is_prejepa:
            encode_kwargs['emb_keys'] = []

        embs = []
        with torch.no_grad():
            for i in range(0, len(frames), chunk_size):
                info = {"pixels": frames[i:i + chunk_size].unsqueeze(0).to(self._device)}
                self._model.encode(info, **encode_kwargs)
                emb = info["emb"].squeeze(0)
                embs.append(emb.cpu())
        return torch.cat(embs).to(self._device)

    # -- Solver protocol -------------------------------------------------------

    def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
        self._physical_action_dim = int(np.prod(action_space.shape[1:]))
        self._action_block = config.action_block
        self._n_envs = n_envs

        # Fallback: single reference if set_reference_episodes wasn't called
        if not self._references:
            ref_cfg = self._reference_cfg
            embs = self._encode_episode(
                dataset_name=ref_cfg.dataset_name,
                episode_idx=ref_cfg.get("episode_idx", 0),
                start_step=0,
                end_step=None,
                chunk_size=ref_cfg.get("chunk_size", 64),
                cache_dir=ref_cfg.get("cache_dir", None),
            )
            self._references = [embs]
            self._env_ref_indices = np.zeros(n_envs, dtype=np.int64)

        # Optional per-view weighted L2 for per-view concat latents:
        #   cost_weights.distance:     "l2_view_weighted"
        #   cost_weights.view_dims:    [192, 192]      (AW) or [192, 192, 192] (AWP)
        #   cost_weights.view_weights: [1.0, 3.0]      (wrist 3x, agent 1x)
        # Both view_dims / view_weights default to None → plain L2 (SV behavior).
        view_dims = self._cost_weights.get("view_dims", None)
        view_weights = self._cost_weights.get("view_weights", None)
        if view_dims is not None:
            view_dims = list(view_dims)
        if view_weights is not None:
            view_weights = [float(w) for w in view_weights]

        # Optional ``ref_stride`` (default action_block). When set ≠ action_block
        # (e.g., ref_stride = action_block / c), MPC tracks a faster reference
        # by jumping more frames per planning step. ``advance_steps`` scales
        # the per-env-step ref-step advance by ``ref_stride / action_block``
        # so the ref window also progresses faster between replans.
        ref_stride_cfg = self._cost_weights.get("ref_stride", None)
        self._ref_stride = (
            float(ref_stride_cfg) if ref_stride_cfg is not None
            else float(config.action_block)
        )

        self.cost_model = self.COST_MODEL_CLS(
            world_model=self._model,
            references=self._references,
            q_tracking=self._cost_weights.tracking,
            q_terminal=self._cost_weights.terminal,
            q_action=self._cost_weights.action,
            q_action_smooth=self._cost_weights.get("action_smooth", 0.0),
            action_block=config.action_block,
            distance=self._cost_weights.distance,
            view_dims=view_dims,
            view_weights=view_weights,
            ref_stride=self._ref_stride,
            z_norm=self._cost_weights.get("z_norm", False),
            horizon_weight=self._cost_weights.get("horizon_weight", "uniform"),
        ).to(self._device)

        # Density-proxy OOD penalty (optional). Loaded once; per-solve state set
        # in solve(). Disabled by default → bit-exact baseline.
        dens_cfg = self._cost_weights.get("density", None)
        if dens_cfg is not None and dens_cfg.get("enable", False):
            from latent_modules.mpc.density import DensityProxy
            dp = DensityProxy(dens_cfg.npz, dens_cfg.meta, self._device)
            if dens_cfg.get("q", None) is not None:
                dp.q = float(dens_cfg.q)
            self.cost_model._density_proxy = dp
            self.cost_model._density_lambda = float(dens_cfg.get("lambda", 1.0))
            print(f"[MPC] density proxy ENABLED  lambda={self.cost_model._density_lambda} "
                  f"q={dp.q:.3f}  k={dp.k}  ref={dp.ref_zpca.shape[0]}")

        # Ensemble-JRD OOD penalty (UNISafe port). Disabled by default.
        ens_cfg = self._cost_weights.get("ensemble", None)
        if ens_cfg is not None and ens_cfg.get("enable", False):
            from latent_modules.mpc.ensemble_proxy import EnsembleProxy
            ep = EnsembleProxy(ens_cfg.ckpt, ens_cfg.meta, self._device)
            if ens_cfg.get("eps", None) is not None:
                ep.eps = float(ens_cfg.eps)
            self.cost_model._ensemble_proxy = ep
            self.cost_model._ensemble_lambda = float(ens_cfg.get("lambda", 1.0))
            print(f"[MPC] ensemble proxy ENABLED  lambda={self.cost_model._ensemble_lambda} "
                  f"eps={ep.eps:.3f}  K={ep.ens.ensemble_size}")

        # Initialize per-env step counters. float64 to allow non-integer
        # advance per env-step when ref_stride != action_block; cast to long
        # at solve() time before injection into info_dict.
        self._env_ref_steps = np.zeros(n_envs, dtype=np.float64)

        # Inner CEM solver (no augmented action space — just physical)
        self._inner_solver = hydra.utils.instantiate(
            self._inner_solver_cfg, model=self.cost_model
        )
        self._inner_solver.configure(
            action_space=action_space,
            n_envs=n_envs,
            config=swm.PlanConfig(
                horizon=config.horizon,
                receding_horizon=config.receding_horizon,
                action_block=config.action_block,
            ),
        )

    def advance_steps(self, n: int = 1):
        """Advance reference step counters (called after each env step).

        With ``ref_stride != action_block``, the ref advances faster (or
        slower) than the env: ``Δref = n · ref_stride / action_block``. Float
        accumulation; cost.py casts to long at gather time.
        """
        self._env_ref_steps += n * (self._ref_stride / self._action_block)

    def reset_steps(self):
        """Reset all step counters to 0."""
        self._env_ref_steps = np.zeros(self._n_envs, dtype=np.float64)

    def _pre_encode(self, info_dict: dict, batch_size: int | None = None) -> torch.Tensor:
        """Encode (pixels + MV streams) once per solve, with optional cache reuse.

        Diffusion-MPC policies encode the current frame every env-step for
        ring-buffer maintenance. At replan boundaries the solver would
        normally re-encode the same frame, wasting one ViT forward. The
        policy can stash its encoded emb on ``self._cached_pre_encoded_emb``;
        this helper consumes it (single-use) when shape matches, otherwise
        falls back to inline encode.

        For GT-MPC (no DiffusionMPCPolicy), the cache is never set so the
        fallback path runs unchanged — bit-perfect parity with prior behavior.

        Args:
            info_dict: solver input dict with at least ``pixels``.
            batch_size: expected batch dim (= n_envs). Used to validate cache
                shape. If None, validation is skipped.
        Returns:
            ``(B, T_hist, D)`` emb on solver device.
        """
        cached = getattr(self, "_cached_pre_encoded_emb", None)
        if cached is not None and (batch_size is None or cached.shape[0] == batch_size):
            self._cached_pre_encoded_emb = None  # consume — single-use
            return cached
        with torch.no_grad():
            encode_input = {
                "pixels": info_dict["pixels"].to(self._device, non_blocking=True)
            }
            for _key in ("wrist_pixels", "proprio_gripper_opening",
                         "proprio_effector_pos"):
                if _key in info_dict and torch.is_tensor(info_dict[_key]):
                    encode_input[_key] = info_dict[_key].to(self._device, non_blocking=True)
            self._model.encode(encode_input)
            return encode_input["emb"]

    def refresh_references(self, new_refs: list[torch.Tensor]) -> None:
        """In-place replace cost_model._refs_padded with fresh references.

        Used by diffusion-MPC: at each replan, sample new latent refs from
        the diffusion planner and hot-swap them into the cost model. The
        ``_refs_padded`` register_buffer is NOT auto-tracked from
        ``self._references``, so it must be rebuilt manually here.

        Caller is responsible for resetting ``self._env_ref_steps[:] = 0``
        after this call (new refs always start at index 0).

        Args:
            new_refs: list of (T_ref, D) tensors, one per (env, candidate)
                pair. Length must match cost_model's allocated batch dim
                (n_envs for plain MPC, n_envs * n_ensemble for ensemble).
        """
        if not new_refs:
            raise ValueError("refresh_references: new_refs is empty")
        self._references = list(new_refs)
        padded = pad_sequence([r.detach().cpu() for r in new_refs], batch_first=True)
        lengths = torch.tensor([r.shape[0] for r in new_refs], dtype=torch.long)
        cost_model = self.cost_model
        device = cost_model._refs_padded.device
        # Re-register handles shape changes; persistent=False mirrors __init__.
        cost_model.register_buffer("_refs_padded", padded.to(device), persistent=False)
        cost_model.register_buffer("_ref_lengths", lengths.to(device), persistent=False)

    @property
    def action_dim(self) -> int:
        return self._physical_action_dim

    @property
    def n_envs(self) -> int:
        return self._inner_solver.n_envs

    @property
    def horizon(self) -> int:
        return self._inner_solver.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        # GT-init override: if a pre-recorded action sequence is available
        # (eval.init_action_npz), use the next slice as init_action. Falls
        # through silently when list is exhausted or unset.
        if (init_action is None
                and self._gt_init_actions is not None
                and self._gt_solve_count < self._gt_init_actions.shape[0]):
            init_action = self._gt_init_actions[self._gt_solve_count].to(self._device)
            self._gt_solve_count += 1

        # Inject per-env reference index and current step
        if self._env_ref_indices is not None:
            info_dict["_ref_idx"] = torch.from_numpy(self._env_ref_indices).long()
        if self._env_ref_steps is not None:
            info_dict["_ref_step"] = torch.from_numpy(self._env_ref_steps).long()

        # Pre-encode pixels (and any MV view tensors) ONCE per solve so the
        # encoder doesn't run on every CEM iteration. CEMSolver expands
        # info_dict tensors to (B, S, ...) before calling cost_model, so the
        # cached emb (B, T_hist, D) becomes (B, S, T_hist, D) which
        # JEPA.rollout's use_precomputed_emb branch picks up. Diffusion-MPC
        # may have already encoded the current frame (for ring buffer) and
        # stashed it on ``self._cached_pre_encoded_emb``; ``_pre_encode``
        # consumes that single-use cache to skip the redundant ViT forward.
        info_dict["emb"] = self._pre_encode(
            info_dict, batch_size=info_dict["pixels"].shape[0]
        )
        # current emb snapshot (last history frame) — feeds the density-proxy
        # state and the dist-shift cur_emb dump.
        _cur_emb = info_dict["emb"]

        # [density-proxy] per-solve local corpus-action distribution from the
        # current state z (last history frame) → get_cost penalises OOD actions.
        if getattr(self.cost_model, "_density_proxy", None) is not None:
            self.cost_model._density_proxy.set_state(_cur_emb[:, -1, :])

        outputs = self._inner_solver.solve(info_dict, init_action=init_action)

        # [dist-shift ablation] record cur_emb (deployment latent) per solve →
        # eval dumps these to rollout_latents.npz for the WM↔deployment W1 gap.
        try:
            self._cur_emb_history.append(
                _cur_emb[:, -1, :].detach().to("cpu").float().numpy().copy())
        except Exception as _e:
            print(f"[dist-shift] cur_emb logging skipped: {_e!r}")

        # Per-solve log accumulator — picked up by eval entry on save_data=true.
        # Stores elite-mean action plan, init/final cost+l2+|a|, and the
        # per-env ref_step at this solve. ν fields stay None for MPC.
        self._solve_log.append({
            "ref_step":   self._env_ref_steps.copy() if self._env_ref_steps is not None else None,
            "actions":    outputs["actions"].clone(),  # (n_envs, H, A)
            "costs":      list(outputs.get("costs", [])),
            "init_l2":    list(outputs.get("init_l2", [])),
            "final_l2":   list(outputs.get("final_l2", [])),
            "init_aabs":  list(outputs.get("init_aabs", [])),
            "final_aabs": list(outputs.get("final_aabs", [])),
            "init_nu":    list(outputs.get("init_nu", [])),
            "final_nu":   list(outputs.get("final_nu", [])),
        })
        return outputs


class MPCSolverDebug(MPCSolver):
    """MPCSolver wired to :class:`MPCCostModelDebug` for debug visualization.

    Clears accumulated per-env breakdowns at the top of every solve and
    exposes them via :meth:`get_env_breakdowns` for downstream consumers
    (see ``latent_modules.mpc.debug_video.DebugLogger``).

    Select via Hydra: ``solver=mpc_debug`` (SV) or ``solver=mpc_multiview_debug``
    (MV). Debug video is only rendered when ``eval.debug_video=true``;
    otherwise the collected breakdowns are simply unused.
    """

    COST_MODEL_CLS = MPCCostModelDebug

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        # Fresh per-env breakdown dict each solve so stale batch data
        # doesn't leak into the new debug frame.
        self.cost_model._env_breakdowns.clear()
        return super().solve(info_dict, init_action=init_action)

    def get_env_breakdowns(self) -> dict[int, dict]:
        """Per-env cost breakdowns keyed by ref_idx, from the last solve()."""
        return self.cost_model._env_breakdowns
