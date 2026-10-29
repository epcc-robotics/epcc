"""Task-orthogonal WM-Full policy — CEM in world model, terminal-only cost.

For cube: uses ``_setup_eval_solver`` from ``eval_mpc_multiview`` (the MPC
variant, NOT MPCC) with ``MultiviewWorldModelPolicy`` and terminal-cost
weights. Same wrap-and-delegate pattern as MPCCPolicy + DPPolicy.

For PushT: legacy uses ``eval_full.py`` with ``policy=wm_goal_frame`` — a
substantially different stack. Day 5 covers cube only; PushT WM-Full
remains via the Day 1/2 dispatcher until a follow-up iteration.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig

from policy.base import SolverPolicy


class WMFullPolicy(SolverPolicy):
    """CEM-in-WM policy. Wraps legacy MultiviewWorldModelPolicy (cube) or
    swm.policy.WorldModelPolicy (PushT) via eval_mpc_multiview._setup_eval_solver
    (MPC variant, NOT MPCC).

    Construction kwargs (all optional, sensible defaults):
      legacy_policy_cls : str | None — fully-qualified class name for the inner
        policy. ``None`` → MultiviewWorldModelPolicy (cube). Pass
        ``stable_worldmodel.policy.WorldModelPolicy`` for PushT.
    """

    def __init__(
        self,
        legacy_policy_cls: str | None = None,
        goal_frame_only: bool = False,
        goal_frame_mode: str = "last",
        **kwargs: Any,
    ) -> None:
        """``goal_frame_only`` (default False = published behaviour):

        - False: **moving-window** terminal cost. ``eval_unified`` advances the
          solver's ``_env_ref_steps`` once per env-step, so the terminal target is
          the demo frame ``ref_step + horizon·action_block`` ahead of the current
          time and only clamps to the demo's last frame in the final steps. The
          policy therefore tracks a demo waypoint, not a goal image.
        ``goal_frame_mode`` (only read when ``goal_frame_only=True``):

        - ``"last"``: the goal latent is the demo's LAST frame. In these corpora that
          frame lies inside a post-placement tail (cube: ~60% of the episode, arm
          retreated 0.13-0.16 m, zero actions; PushT: agent parked 24-136 px away), so
          the target mixes "object at goal" with "agent parked away".
        - ``"settle"``: the goal latent is the frame where the OBJECT first reaches the
          goal and stays there — the placement instant, matching the scorers' own
          predicates (PushT: |block-goal| < 20 px and |angle| < pi/9, the park_metrics
          tolerances; cube: |cube-target| < 0.07 m, the eval's goal_dist_thresh). The
          arm is still at the object there, so the goal image is the task being
          completed rather than the demonstrator's parking manoeuvre.

        - True: **pure goal-frame** cost. ``_env_ref_steps`` is pinned at
          ``ref_len - 1`` (the demo's last frame = the goal image) before every
          action, so the terminal cost is always ``d(ẑ_{T-1}, z_goal)`` regardless
          of episode time. Re-pinned in ``get_action`` so it is immune to the
          eval loop's ``advance_steps`` and to any ``reset_steps`` ordering.
          This is the port of the legacy
          ``legacy/pusht_baseline/configs/policy/wm_goal_frame.yaml``
          (``WMTerminalPolicy(goal_frame_only=True)``).

        Nothing else changes: same world model, scaler, CEM, clamps and cost
        weights — only which reference frame the terminal cost points at.
        """
        super().__init__(**kwargs)
        self._legacy_policy_cls_path = legacy_policy_cls
        self.goal_frame_only = bool(goal_frame_only)
        if goal_frame_mode not in ("last", "settle"):
            raise ValueError(f"goal_frame_mode must be 'last' or 'settle', got {goal_frame_mode!r}")
        self.goal_frame_mode = goal_frame_mode
        self._goal_ref_steps = None      # (n_envs,) float64 reference-cursor index per env
        self._inner = None
        self._solver = None
        self._model = None
        self._n_replans = 0

    def reset(self, world: Any, cfg: DictConfig | None = None, eval_world: Any | None = None) -> None:
        if cfg is None or eval_world is None:
            raise ValueError("WMFullPolicy.reset requires both cfg and eval_world")

        from omegaconf import OmegaConf
        from eval_mpc_multiview import _setup_eval_solver, MultiviewWorldModelPolicy

        # Resolve inner policy class (cube default = MultiviewWorldModelPolicy).
        if self._legacy_policy_cls_path is None:
            policy_cls = MultiviewWorldModelPolicy
        else:
            mod_name, _, cls_name = self._legacy_policy_cls_path.rpartition(".")
            import importlib
            policy_cls = getattr(importlib.import_module(mod_name), cls_name)

        legacy_policy_group = str(cfg.task.policy_config_group)
        OmegaConf.set_struct(cfg, False)
        saved_policy = cfg.policy
        cfg.policy = legacy_policy_group
        try:
            _model, _solver, inner_policy, _plan_config = _setup_eval_solver(
                cfg, eval_world,
                gt_refs=True,
                policy_cls=policy_cls,
                policy_kwargs={},
            )
        finally:
            cfg.policy = saved_policy
            OmegaConf.set_struct(cfg, True)

        # HF ViT compat patch.
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

        self._model = _model
        self._solver = _solver
        self._inner = inner_policy
        self._inner.set_env(world.envs)

        # Goal-frame-only: pin the reference cursor at the demo's LAST frame.
        # cost.py clamps ref_indices to ref_len-1, so the whole horizon collapses
        # onto the goal latent.
        if self.goal_frame_only:
            import numpy as _np
            refs = getattr(self._solver, "_references", None)
            if not refs:
                raise RuntimeError("goal_frame_only=True but the solver has no references")
            ref_lengths = _np.array([len(r) for r in refs], dtype=_np.int64)
            ref_idx = _np.asarray(self._solver._env_ref_indices)
            last_steps = (ref_lengths[ref_idx] - 1).astype(_np.float64)
            if self.goal_frame_mode == "settle":
                self._goal_ref_steps = self._settle_ref_steps(eval_world, last_steps)
            else:
                self._goal_ref_steps = last_steps
            self._solver._env_ref_steps = self._goal_ref_steps.copy()
            print(f"[wm-full] goal_frame_only=True mode={self.goal_frame_mode}: pinned ref_steps to "
                  f"{self._goal_ref_steps[:5].astype(int).tolist()}... "
                  f"(n_envs={len(self._goal_ref_steps)}, ref_len min/max="
                  f"{int(ref_lengths.min())}/{int(ref_lengths.max())})")

    def _settle_ref_steps(self, eval_world: Any, last_steps: "np.ndarray") -> "np.ndarray":
        """Reference-cursor index of the placement instant, one per env.

        Reference ``r`` corresponds to demo frame ``start + r`` (the solver builds it as
        ``[start, start + goal_offset]``), so the cursor is ``settle_frame - start``,
        clamped into the reference. ``settle_frame`` is the earliest frame from which the
        object stays in tolerance until the end of the demo — the same predicates the
        scorers use. Falls back to the last frame for any episode that never settles
        (and says so), so one odd demo cannot silently change the whole cell.
        """
        import numpy as np
        import stable_worldmodel as swm

        POS, ANG, CUBE_D = 20.0, np.pi / 9.0, 0.07
        ref_cfg = self._solver._reference_cfg
        ds_name = str(ref_cfg.dataset_name)
        kwargs = {"name": ds_name, "frameskip": 1, "keys_to_load": []}
        cache_dir = ref_cfg.get("cache_dir", None)
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        ds = swm.data.HDF5Dataset(**kwargs)
        eps = np.asarray(eval_world.eval_episodes, dtype=np.int64)
        starts = np.asarray(eval_world.eval_start_idx, dtype=np.int64)

        def final_run_start(ok: np.ndarray) -> int:
            """Earliest frame from which ``ok`` stays True to the end (-1 if never)."""
            if not bool(ok[-1]):
                return -1
            i = len(ok) - 1
            while i > 0 and bool(ok[i - 1]):
                i -= 1
            return int(i)

        out, n_fallback, fracs = [], 0, []
        for e, st in zip(eps, starts):
            ep = ds.load_episode(int(e))
            keys = set(ep.keys())
            if "privileged_block_0_pos" in keys:                      # cube
                cube = np.asarray(ep["privileged_block_0_pos"], np.float64)[:, :3]
                tgt = None
                for k in ("goal_privileged_target_block_pos", "privileged_target_block_pos",
                          "goal_privileged_block_0_pos"):
                    if k in keys:
                        tgt = np.asarray(ep[k], np.float64); break
                if tgt is None:
                    raise RuntimeError(f"no target column in {ds_name}; have {sorted(keys)[:20]}")
                tg = tgt[-1, :3] if tgt.ndim == 2 else tgt[:3]
                ok = np.linalg.norm(cube - tg, axis=-1) < CUBE_D
            elif "block_pose" in keys and "goal_pose" in keys:        # PushT / slippery
                blk = np.asarray(ep["block_pose"], np.float64)
                goal = np.asarray(ep["goal_pose"], np.float64)
                goal = goal[-1] if goal.ndim == 2 else goal
                d = np.abs(blk[:, 2] - goal[2])
                ang = np.minimum(d, 2 * np.pi - d)
                ok = (np.linalg.norm(blk[:, :2] - goal[:2], axis=-1) < POS) & (ang < ANG)
            else:
                raise RuntimeError(f"{ds_name}: no block_pose/goal_pose nor privileged_block_0_pos "
                                   f"columns to define the placement instant (have {sorted(keys)[:20]})")
            f = final_run_start(ok)
            if f < 0:
                n_fallback += 1
                out.append(None)
            else:
                out.append(float(max(f - int(st), 0)))
                fracs.append((f - int(st)) / max(len(ok) - int(st) - 1, 1))
        steps = np.array([last_steps[i] if v is None else min(v, last_steps[i])
                          for i, v in enumerate(out)], dtype=np.float64)
        print(f"[wm-full] goal_frame_mode=settle: placement instant found for "
              f"{len(eps) - n_fallback}/{len(eps)} episodes"
              + (f" ({n_fallback} never settle -> last frame)" if n_fallback else "")
              + (f"; cursor is at {np.median(fracs):.2f} of the reference (median), "
                 f"vs 1.00 for mode=last" if fracs else ""))
        return steps

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        if self._inner is None:
            raise RuntimeError("WMFullPolicy.get_action called before reset(...)")
        # WM-Full is chunked-replan too: inner WorldModelPolicy._action_buffer
        # holds receding_horizon × action_block actions; CEM-in-WM fires (via
        # solver.solve) only when the buffer is empty. Detect the actual
        # replan by sampling the buffer state before delegation (mirrors
        # MPCCPolicy / DTGMPCCPolicy).
        if self.goal_frame_only and self._goal_ref_steps is not None:
            # Re-pin every step: the eval loop calls solver.advance_steps() after
            # each world.step(), which would otherwise walk the cursor past the
            # goal (harmless thanks to the clamp, but this keeps the invariant
            # explicit and survives any reset_steps call).
            self._solver._env_ref_steps = self._goal_ref_steps.copy()
        buf = getattr(self._inner, "_action_buffer", None)
        if buf is None or len(buf) == 0:
            self._n_replans += 1
        return self._inner.get_action(info_dict, **kwargs)

    @property
    def n_replans(self) -> int:
        """Total CEM-in-WM solves since reset (one per buffer refill)."""
        return int(self._n_replans)
