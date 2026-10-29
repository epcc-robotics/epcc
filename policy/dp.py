"""Task-orthogonal Diffusion Policy (CleanDiffuser backend).

Wraps the legacy ``CleanDiffuserDPPolicy`` from ``eval_cleandiffuser_dp.py``
in a ``TaskOrthogonalPolicy`` interface. The legacy class already implements
chunked replanning + multiview-pixel handling correctly; we don't reimplement
that, we wrap it so the unified loop can call ``policy.get_action(info)`` once
per step and trust the inner class to manage its action queue.

Day 3 design: this is a wrapper, not a from-scratch port. Reproduction
guarantee comes from delegating to the byte-equivalent legacy class.
Day 4+ can refactor chunked-replan logic into ``ChunkedReplanPolicy`` once
the patterns across DP / DTG variants are clear.

Construction is two-phase to match how the legacy harness builds the policy:
  1. ``__init__`` stores config (ckpt path, hyperparams) — cheap, no GPU.
  2. ``reset(world)`` builds the planner + inner policy — needs the world
     to attach the policy via ``world.set_policy(self)``.

The unified loop in ``scripts/baseline_v2/eval_unified.py`` will call
``reset(world)`` once after composing the world. Same lifecycle for every
TaskOrthogonalPolicy subclass.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig

from policy.base import TaskOrthogonalPolicy


class DPPolicy(TaskOrthogonalPolicy):
    """CleanDiffuser-DP wrapped as a TaskOrthogonalPolicy.

    Construction kwargs (from Hydra recipe ``policy/dp_inprocess.yaml``):
      ckpt_path        : str — path to CleanDiffuser DP checkpoint .pt
      train_h5_path    : str — path to training H5 (for action normalizer)
      action_dim       : int — env action dim (cube=5, pusht=2)
      action_horizon   : int — diffusion chunk length (default 16)
      action_steps     : int — actions consumed per replan (default 8)
      obs_steps        : int — history frames for obs (default 2)
      image_size       : int — input image side (default 96)
      n_views          : int — number of camera views (cube=2, pusht=1)
      crop_shape       : list[int] — center crop (default [84, 84])
      sample_steps     : int — diffusion sampler steps (default 50)
      use_group_norm   : bool — backbone normalization (default True)
      use_seq          : bool — sequence-style backbone (default True)
      nn               : str — backbone name (default "chi_unet")
    """

    def __init__(
        self,
        ckpt_path: str,
        train_h5_path: str,
        action_dim: int,
        action_horizon: int = 16,
        action_steps: int = 8,
        obs_steps: int = 2,
        image_size: int = 96,
        n_views: int = 2,
        crop_shape: list[int] | None = None,
        sample_steps: int = 50,
        use_group_norm: bool = True,
        use_seq: bool = True,
        nn: str = "chi_unet",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._ckpt_path = str(ckpt_path)
        self._train_h5_path = str(train_h5_path)
        self._action_dim = int(action_dim)
        self._action_horizon = int(action_horizon)
        self._action_steps = int(action_steps)
        self._obs_steps = int(obs_steps)
        self._image_size = int(image_size)
        self._n_views = int(n_views)
        self._crop_shape = list(crop_shape) if crop_shape is not None else [84, 84]
        self._sample_steps = int(sample_steps)
        self._use_group_norm = bool(use_group_norm)
        self._use_seq = bool(use_seq)
        self._nn = str(nn)
        self._inner = None  # built in reset(world)

    def reset(self, world: Any, cfg: DictConfig | None = None, eval_world: Any | None = None) -> None:
        """Build the planner + inner CleanDiffuserDPPolicy.

        Defers the heavy build (model load, planner init) until the world is
        available. ``cfg`` is the full Hydra cfg; legacy helpers
        (``build_cleandiffuser_dp_planner``, ``_setup_eval_solver_mpcc``) read
        many keys from it.

        cfg.policy clash: legacy ``AutoCostModel(cfg.policy: str)`` expects a
        config-group string name. Our v2 layer uses ``cfg.policy`` as the
        TaskOrthogonalPolicy instantiation DictConfig. We monkey-patch
        cfg.policy with the legacy string from ``cfg.task.policy_config_group``
        for the duration of the legacy helper call, then restore it.
        """
        if cfg is None:
            raise ValueError("DPPolicy.reset requires cfg=cfg argument")

        from omegaconf import OmegaConf
        from eval_cleandiffuser_dp import (
            CleanDiffuserDPPolicy,
            build_cleandiffuser_dp_planner,
        )
        from eval_mpcc_multiview import _setup_eval_solver_mpcc

        legacy_policy_group = str(cfg.task.policy_config_group)
        OmegaConf.set_struct(cfg, False)
        saved_policy = cfg.policy
        cfg.policy = legacy_policy_group
        try:
            dp = build_cleandiffuser_dp_planner(
                ckpt_path=self._ckpt_path,
                train_h5_path=self._train_h5_path,
                cfg=cfg,
                device="cuda",
            )

            ew = eval_world if eval_world is not None else type(
                "_NS", (), {"world": world, "n_eval": world.num_envs}
            )()
            _model, _solver, inner_policy, _plan_config = _setup_eval_solver_mpcc(
                cfg, ew,
                gt_refs=False,
                policy_cls=CleanDiffuserDPPolicy,
                policy_kwargs={
                    "dp": dp,
                    "action_block": int(cfg.plan_config.action_block),
                    "receding_horizon": int(cfg.plan_config.receding_horizon),
                    "dump_dir": None,  # disable replan dumps in v2 path
                },
            )
        finally:
            cfg.policy = saved_policy
            OmegaConf.set_struct(cfg, True)

        # HF ViT compat patch — mirrors eval_cleandiffuser_dp.run() ViT shim.
        import transformers as _hf_transformers
        import torch
        for _m in _model.modules():
            if hasattr(_m, "config") and isinstance(_m.config, _hf_transformers.PretrainedConfig):
                _fresh = type(_m.config)()
                for _k, _v in _fresh.__dict__.items():
                    if _k not in _m.config.__dict__:
                        setattr(_m.config, _k, _v)
            if "ViTSelfAttention" in type(_m).__name__ and not hasattr(_m, "dropout"):
                _m.dropout = torch.nn.Dropout(getattr(_m, "dropout_prob", 0.0))

        self._inner = inner_policy
        self._dp = dp
        self._solver = _solver
        self._model = _model

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        if self._inner is None:
            raise RuntimeError(
                "DPPolicy.get_action called before reset(world, cfg). "
                "The unified loop must call reset() once after world.set_policy(self)."
            )
        return self._inner.get_action(info_dict, **kwargs)

    # Expose inner counters for debug/metrics.
    @property
    def n_replans(self) -> int:
        return int(getattr(self._inner, "_n_replans", 0)) if self._inner else 0

