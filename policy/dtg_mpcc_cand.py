"""EPCC-5 — DTG-MPCC with a predefined-operator candidate library.

Extends :class:`policy.dtg_mpcc.DTGMPCCPolicy` with the machinery for
§V-D of the paper: at every replan a frozen policy chunk (and/or an
inverse-dynamics decoding of the DTG contour) is expanded into a library of
rate-``k`` accumulate-downsampled candidates, each paired with the progress
schedule that rate implies, and the world model scores them.

Rows this class serves (``candidates.mode``):
  ``select``     arg-min of the library, executed as-is (no CEM).
  ``refine``     arg-min seeds the CEM mean; ν pinned (recipe sets nu_var_scale=0).
  ``full_cand``  arg-min seeds the CEM mean; ν free.  Internal control only.
The paper's *Full EPCC* row does not use this class at all — it is the stock
``DTGMPCCPolicy``.

Two facts make the schedule exact rather than heuristic:
  * ``ReferenceTrajectory.theta[i]`` already stores the arc-length coordinate of
    DTG waypoint ``i``, and one waypoint spans ``stride`` demo env-steps while
    one CEM step spans ``action_block`` executed env-steps.  A rate-``k`` chunk
    therefore advances ``k · action_block / stride`` waypoints per CEM step, so
    ``S_j = theta[j·k·action_block/stride]`` — exactly ``theta[j·k]`` for the
    H16/K2/stride-5 generation (stride == action_block == 5) and ``theta[5·j·k]``
    for the H84/K6/stride-1 generators (2026-09-07). ``reset`` refuses a
    generator whose stride does not divide ``action_block``.
  * Converting that back through ``nu_scale = action_block/(c·stride·N)`` gives
    ``ν ≈ k·c`` — and the solver config already defines ``ν = c`` as expert
    pace.  No new calibration constant is introduced.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from policy.dtg_mpcc import DTGMPCCPolicy


class DTGMPCCCandidatePolicy(DTGMPCCPolicy):
    """DTG-MPCC + EPCC-5 candidate library (families P = policy, R = reference)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cc: DictConfig | None = None
        self._dp = None                  # CleanDiffuserDPPlanner (family P)
        self._dp_ring: torch.Tensor | None = None
        self._invdyn = None              # InvDynMLP (family R)
        self._scaler = None              # sklearn StandardScaler on env actions
        self._log_fp = None
        self._replan_idx = 0
        self._env_step = 0
        self._arrays: dict[str, list] = {}
        self._outdir: Path | None = None

    # ------------------------------------------------------------------ setup
    def reset(self, world: Any, cfg: DictConfig | None = None, eval_world: Any | None = None) -> None:
        super().reset(world, cfg=cfg, eval_world=eval_world)

        cc = cfg.get("candidates", None)
        if cc is None:
            raise ValueError("DTGMPCCCandidatePolicy requires a cfg.candidates block")
        self._cc = cc
        self._mode = str(cc.get("mode", "select"))
        self._families = str(cc.get("library", "P")).upper()   # "P" | "R" | "PR"
        self._rates = [float(r) for r in cc.get("rates", [1.0, 1.5, 2.0, 2.5, 3.0])]
        self._n_dp = int(cc.get("n_dp_samples", 8))
        self._dp_chunk_len = int(cc.get("dp_chunk_len", 15))
        self._tf_mode = str(cc.get("transform", "accumulate"))
        self._base_seed = int(cfg.get("seed", 42))

        self._action_block = int(cfg.plan_config.action_block)
        self._horizon = int(cfg.plan_config.horizon)

        # Waypoints the contour advances per CEM step (see _schedules). The generator's
        # stride is read off the checkpoint, not the yaml: policy/dtg_mpcc.py only syncs
        # solver.reference.stride when the yaml says None/1, so the yaml can lie.
        gen_stride = int(self._inner._diff.stride)
        if self._action_block % gen_stride != 0:
            raise ValueError(
                f"action_block ({self._action_block}) must be a multiple of the generator "
                f"stride ({gen_stride}) for the candidate schedule "
                "S_j = theta[j*k*action_block/stride] to land on waypoints at integer k")
        self._wp_per_cem_step = self._action_block // gen_stride
        # The MPCC cost model must have been built at the same stride, or nu_scale
        # (= action_block / (c * stride * N)) is off by their ratio with no error anywhere.
        ref_stride = getattr(self._solver.cost_model, "stride", None)
        if ref_stride is not None and int(ref_stride) != gen_stride:
            raise ValueError(
                f"solver.reference.stride ({int(ref_stride)}) != generator stride ({gen_stride}); "
                "pass ++solver.reference.stride=1 (or leave it unset) so policy/dtg_mpcc.py "
                "syncs it to the checkpoint — nu_scale would otherwise be scaled by "
                f"{int(ref_stride) / gen_stride:g}")

        # Env-action scaler (StandardScaler on the WM's own corpus). Same object
        # the CEM uses for env-space projection, so candidates and CEM samples
        # live in one normalized frame.
        self._scaler = self._inner.process["action"]

        if "P" in self._families:
            self._dp = self._build_dp(cfg)

        # mode="refine_sup": SuP's own scheduler picks the rate and we refine the actions
        # at it. The gate is SuP's, unmodified -- we only read its argmax. Its Q expects
        # (state, chunk) at the chunk length it was trained on, so the IQL must have been
        # built at dp_chunk_len; a mismatch is silently zero-padded inside the gate
        # (sup_state_gate.py:142-152), which would gate a different policy than we run.
        self._gate = None
        sup_cfg = cfg.candidates.get("sup", None)
        if self._mode == "refine_sup":
            # The gate scores a DP action chunk, so family P has to be in the library.
            if "P" not in self._families:
                raise ValueError("mode='refine_sup' needs library P — the gate scores "
                                 f"the DP chunk, but library is {self._families!r}")
            if sup_cfg is None:
                raise ValueError("mode='refine_sup' needs a candidates.sup block "
                                 "(adm_path / iql_path / channel)")
            from policy.sup_state_gate import StateADMGate
            self._gate = StateADMGate(
                task_name=str(cfg.task.get("task_name", cfg.task.get("name", ""))),
                adm_path=str(sup_cfg.adm_path),
                k_values=[int(k) for k in self._rates],   # superseded by the ckpt's KS
                eps=float(sup_cfg.get("eps", 0.0)),
                downsample_mode=str(sup_cfg.get("downsample_mode", self._tf_mode)),
                channel=str(sup_cfg.get("channel", "full")),
                iql_path=str(sup_cfg.iql_path),
                device="cuda",
            )
            exp = int(self._gate.q_sm.shape[0])
            got = len(self._gate.cols) + self._dp_chunk_len * self._dp.action_dim
            if exp != got:
                raise ValueError(
                    f"IQL state dim {exp} != state({len(self._gate.cols)}) + "
                    f"chunk({self._dp_chunk_len}) x adim({self._dp.action_dim}) = {got}. "
                    f"Rebuild the scheduler with --chunk {self._dp_chunk_len}.")
            print(f"[cand] refine_sup: gate KS={self._gate.iql_ks} "
                  f"state_dim={exp} (chunk={self._dp_chunk_len})")
        if "R" in self._families:
            from eval_diffusion_invdyn import _load_invdyn
            ck = cc.get("invdyn_ckpt", None)
            if not ck:
                raise ValueError("library includes R but candidates.invdyn_ckpt is unset")
            self._invdyn = _load_invdyn(str(ck), device="cuda")
            if int(self._invdyn.stride) != gen_stride:
                raise ValueError(
                    f"invdyn.stride ({self._invdyn.stride}) != generator stride "
                    f"({gen_stride}); one waypoint pair must decode to exactly `stride` "
                    "env-actions for S_j = theta[j*k*action_block/stride] to hold."
                )

        inner_cem = getattr(self._solver, "_inner_solver", None)
        if not hasattr(inner_cem, "set_candidates"):
            raise TypeError(
                "candidates.* requires solver.inner_solver._target_="
                "latent_modules.mpcc.solver_select.CandidateCEMSolver "
                f"(got {type(inner_cem).__name__})"
            )
        inner_cem.candidate_builder = self._build_candidates
        self._cem = inner_cem

        outdir = Path(cfg.output.findings_dir) / str(cfg.output.get("logs_subdir", "logs")) / str(cfg.output.run_name)
        outdir.mkdir(parents=True, exist_ok=True)
        self._outdir = outdir
        self._log_fp = open(outdir / "cand_decisions.jsonl", "w")
        # The jsonl truncates on open, but the npz is only rewritten every 20
        # replans (and at exit). A cell that dies before its first flush would
        # otherwise leave a PREVIOUS run's cand_arrays.npz sitting next to a fresh
        # jsonl — and cand_log.py reads meta/pad_frac/valid_cem_steps out of the
        # npz, so the stale table would be read as this run's silently.
        # Delete it up front so "missing" is the failure mode instead of "wrong".
        (outdir / "cand_arrays.npz").unlink(missing_ok=True)
        import atexit
        atexit.register(self.close)          # the eval loop has no teardown hook
        print(f"[cand] mode={self._mode} library={self._families} rates={self._rates} "
              f"n_dp={self._n_dp} scoring={getattr(inner_cem, 'scoring', '?')} "
              f"gen_stride={gen_stride} wp_per_cem_step={self._wp_per_cem_step}")

    def _build_dp(self, cfg: DictConfig):
        """Build ONLY the frozen DP planner.

        Deliberately not ``DPPolicy.reset``: that also constructs a second world
        model + solver (doubling GPU memory), and ``CleanDiffuserDPPolicy``
        truncates its chunk to ``action_block × receding_horizon`` — which is 5
        here, not the 15 the DP/SuP baseline rows execute.  We keep our own ring
        and call the batched sampler directly for the full chunk.
        """
        from eval_cleandiffuser_dp import build_cleandiffuser_dp_planner
        dp_cfg = cfg.candidates.dp
        shim = OmegaConf.create({"diffusion": OmegaConf.to_container(dp_cfg, resolve=True)})
        dp = build_cleandiffuser_dp_planner(
            ckpt_path=str(dp_cfg.ckpt_path),
            train_h5_path=str(dp_cfg.train_h5_path),
            cfg=shim, device="cuda",
        )
        if dp.horizon < self._dp_chunk_len:
            raise ValueError(f"dp.horizon {dp.horizon} < dp_chunk_len {self._dp_chunk_len}")
        return dp

    # ------------------------------------------------------------- env-step
    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        if self._dp is not None:
            from eval_cleandiffuser_dp import _world_pixels_to_cd_tensor
            px = _world_pixels_to_cd_tensor(
                info_dict, use_wrist=self._dp.n_views > 1,
                image_size=self._dp.image_size, device=self._dp.device,
            )
            if self._dp_ring is None:
                self._dp_ring = px.unsqueeze(1).repeat(1, self._dp.K, 1, 1, 1, 1).contiguous()
            else:
                self._dp_ring = torch.roll(self._dp_ring, shifts=-1, dims=1)
                self._dp_ring[:, -1] = px
        out = super().get_action(info_dict, **kwargs)
        self._env_step += 1
        self._log_decision()
        return out

    # ------------------------------------------------------- transform utils
    @staticmethod
    def _accumulate(a: torch.Tensor, k: float, n_out: int) -> torch.Tensor:
        """Rate-``k`` accumulate-downsample of an env-action sequence.

        ``a`` is ``(..., L, A)`` of per-step displacements.  Output element ``j``
        is the displacement swept between fractional source indices ``j·k`` and
        ``(j+1)·k``, i.e. ``C[(j+1)k] − C[jk]`` on the cumulative curve ``C``.
        For integer ``k`` this is exactly ``sum(a[jk:(j+1)k])`` — SuP's
        ``accumulate`` — and it extends the operator to fractional rates
        (retiming) without changing the integer points.
        """
        L = a.shape[-2]
        zero = torch.zeros_like(a[..., :1, :])
        C = torch.cat([zero, a.cumsum(dim=-2)], dim=-2)          # (..., L+1, A)
        idx = torch.arange(n_out + 1, device=a.device, dtype=a.dtype) * k
        idx = idx.clamp(max=float(L))
        lo = idx.floor().long().clamp(max=L)
        hi = (lo + 1).clamp(max=L)
        w = (idx - lo.to(a.dtype)).unsqueeze(-1)
        Cs = C.index_select(-2, lo) * (1 - w) + C.index_select(-2, hi) * w   # (..., n_out+1, A)
        return Cs[..., 1:, :] - Cs[..., :-1, :]

    def _pad_to_plan(self, a: torch.Tensor, n_plan: int) -> torch.Tensor:
        """Constant-velocity extrapolation past the source chunk's reach."""
        L = a.shape[-2]
        if L >= n_plan:
            return a[..., :n_plan, :]
        tail = a[..., -1:, :].expand(*a.shape[:-2], n_plan - L, a.shape[-1])
        return torch.cat([a, tail], dim=-2)

    # ------------------------------------------------------- library builder
    def _build_candidates(self, info_dict: dict, solver: Any):
        """Return ``(candidates, valid_steps, meta)`` for this replan.

        Called from ``CandidateCEMSolver.solve`` — i.e. after the DTG contour for
        this replan has been installed, so ``ref.theta`` is current.
        """
        cost = self._solver.cost_model
        refs = cost.references
        ref_idx = np.asarray(self._solver._env_ref_indices)
        n_envs = len(ref_idx)
        H, ab = self._horizon, self._action_block
        n_plan = H * ab
        device = next(self._solver._model.parameters()).device

        sources: list[tuple[str, torch.Tensor]] = []   # (family, (n_envs, S, L, a_dim))
        if self._dp is not None:
            sources.append(("P", self._sample_dp()))
        if self._invdyn is not None:
            sources.append(("R", self._invdyn_actions(info_dict, refs, ref_idx)))

        self._replan_idx += 1
        cand_env, valid, meta = [], [], []
        for fam, src in sources:
            L = src.shape[-2]
            S = src.shape[1]
            for k in self._rates:
                n_real = max(1, int(np.floor(L / k)))
                a = self._accumulate(src, k, n_out=min(n_real, n_plan))
                a = self._pad_to_plan(a, n_plan)                      # (n_envs, S, n_plan, a_dim)
                cand_env.append(a)
                v = max(1, min(H, n_real // ab))
                valid.extend([v] * S)
                meta.extend([{"family": fam, "k": k, "n_real_env_actions": n_real,
                              "valid_cem_steps": v,
                              "pad_frac": round(1.0 - min(n_real, n_plan) / n_plan, 3)}] * S)
        A_env = torch.cat(cand_env, dim=1)                            # (n_envs, M, n_plan, a_dim)
        M = A_env.shape[1]

        # env space → normalized space (same StandardScaler the CEM projects with)
        mean = torch.as_tensor(self._scaler.mean_, dtype=A_env.dtype, device=A_env.device)
        scale = torch.as_tensor(self._scaler.scale_, dtype=A_env.dtype, device=A_env.device)
        A_norm = (A_env - mean) / scale
        phys = A_norm.reshape(n_envs, M, H, ab * A_env.shape[-1])     # a_dim fastest — matches cost.py

        nu = self._schedules(refs, ref_idx, [m["k"] for m in meta], device)   # (n_envs, M, H)
        cand = torch.cat([phys.to(device), nu.unsqueeze(-1)], dim=-1)
        valid_t = torch.as_tensor(valid, dtype=torch.long, device=device).repeat(n_envs, 1)

        seed_idx = None
        if self._gate is not None:
            # SuP scores the SAME chunk the library was merged from, so its verdict is
            # about the actions we will actually run. n_dp is 1 in this mode, so the
            # (family, rate) pair identifies a candidate uniquely.
            chunk = sources[0][1][:, 0].detach().cpu().numpy()      # (n_envs, L, a_dim)
            Q = self._gate.q_values(info_dict, chunk)               # (n_envs, |KS|)
            ks = np.asarray(self._gate.iql_ks, dtype=float)
            picked_k = ks[np.asarray(Q).argmax(axis=1)]             # (n_envs,)
            by_rate = {m["k"]: i for i, m in enumerate(meta) if m["family"] == "P"}
            rates = np.asarray(sorted(by_rate), dtype=float)
            # A k the library cannot express (a mis-sized scheduler) snaps to the nearest
            # rate we have, and the rate we ran is logged so the substitution is visible.
            snapped = rates[np.abs(rates[None, :] - picked_k[:, None]).argmin(1)]
            seed_idx = np.array([by_rate[float(k)] for k in snapped], dtype=np.int64)
            self._last_sup = {"sup_k": picked_k.tolist(), "sup_k_snapped": snapped.tolist(),
                              "sup_snap_frac": float((snapped != picked_k).mean())}
        return cand, valid_t, meta, seed_idx

    def _sample_dp(self) -> torch.Tensor:
        """``(n_envs, n_dp, dp_chunk_len, a_dim)`` env-space chunks.

        Seeded per replan so every row (SELECT / REFINE / Full(cand-init)) sees
        the same library at the same replan index — DDPM is otherwise unseeded.
        """
        from eval_cleandiffuser_dp import dp_cleandiffuser_sample_actions_batched
        ring = self._dp_ring
        n_envs = ring.shape[0]
        hist = ring.repeat_interleave(self._n_dp, dim=0)               # (n_envs*S, K, V, C, H, W)
        seed = (self._base_seed * 1000003 + self._replan_idx) % (2 ** 31 - 1)
        # Save/restore explicitly rather than fork_rng: the DTG samples from the
        # same global stream, and we must not perturb it relative to the rows
        # that do not build a library.
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            a = dp_cleandiffuser_sample_actions_batched(self._dp, hist)  # (n_envs*S, horizon, A)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        a = a[:, :self._dp_chunk_len, :]
        return a.reshape(n_envs, self._n_dp, self._dp_chunk_len, a.shape[-1])

    def _invdyn_actions(self, info_dict: dict, refs, ref_idx) -> torch.Tensor:
        """``(n_envs, 1, (N-1)·stride, a_dim)`` — InvDyn decoding of the contour.

        One waypoint pair → one ``stride``-long chunk, so the whole contour
        yields ``(N-1)·stride`` env-actions: far longer than the 25-action plan,
        hence family R never needs padding.
        """
        device = next(self._invdyn.parameters()).device
        Z = torch.stack([refs[int(i)].embeddings for i in ref_idx]).to(device)   # (n_envs, N, D)
        n_envs, N, D = Z.shape
        cur = info_dict.get("emb", None)
        if torch.is_tensor(cur):
            cur = cur[:, -1, :].to(device).to(Z.dtype)
        K_inv = int(getattr(self._invdyn, "history_K", 0))
        z_tp = Z[:, 1:, :].reshape(-1, D)
        if K_inv == 0:
            z_t = Z[:, :-1, :].clone()
            if cur is not None:
                z_t[:, 0, :] = cur                    # anchor the first pair on the real latent
            with torch.no_grad():
                chunks = self._invdyn(z_t.reshape(-1, D), z_tp)
        else:
            t_idx = torch.arange(N - 1, device=device)
            offs = torch.arange(K_inv, -1, -1, device=device)
            hist_idx = (t_idx.unsqueeze(1) - offs.unsqueeze(0)).clamp(min=0)
            with torch.no_grad():
                chunks = self._invdyn(Z[:, hist_idx, :].reshape(-1, K_inv + 1, D), z_tp)
        a_dim = int(chunks.shape[-1])
        acts = chunks.view(n_envs, N - 1, int(self._invdyn.stride), a_dim)
        return acts.reshape(n_envs, 1, (N - 1) * int(self._invdyn.stride), a_dim)

    def _schedules(self, refs, ref_idx, rates: list[float], device) -> torch.Tensor:
        """``(n_envs, M, H)`` progress increments ``ν_j = (θ[(j+1)k] − θ[jk]) / nu_scale``.

        Fully vectorised: one gather over the per-env θ table for the whole
        (M, H+1) index grid.  Fractional ``k`` interpolates θ linearly, which is
        the same operation ``ReferenceTrajectory.interpolate`` performs on the
        embedding side, so actions and schedule stay consistent at fractional
        rates.  Note θ is arc length, not time, so the per-step ν varies with
        local contour spacing — ``k·c`` is its mean, by construction.
        """
        H = self._horizon
        E, M = len(ref_idx), len(rates)
        TH = torch.stack([refs[int(i)].theta.to(device).float() for i in ref_idx])   # (E, N)
        N = TH.shape[1]
        NSC = torch.tensor(
            [float(self._solver.cost_model._nu_scales[int(i)]) for i in ref_idx],
            device=device, dtype=torch.float32,
        )                                                                            # (E,)
        k = torch.tensor(rates, device=device, dtype=torch.float32)                  # (M,)
        j = torch.arange(H + 1, device=device, dtype=torch.float32)                  # (H+1,)
        # One CEM step = action_block env-steps = action_block/stride waypoints, so a
        # rate-k candidate reads theta at j*k*(action_block/stride). The factor is 1 for
        # the stride-5 generation (the published ladder is unchanged) and 5 for stride-1.
        idx = (j[None, :] * k[:, None] * float(self._wp_per_cem_step)).clamp(max=float(N - 1))  # (M, H+1)
        lo = idx.floor().long()
        hi = (lo + 1).clamp(max=N - 1)
        w = (idx - lo.float())[None]                                                 # (1, M, H+1)

        THx = TH[:, None, :].expand(E, M, N)
        s_lo = THx.gather(2, lo[None].expand(E, M, H + 1))
        s_hi = THx.gather(2, hi[None].expand(E, M, H + 1))
        theta_q = s_lo * (1 - w) + s_hi * w                                          # (E, M, H+1)
        return ((theta_q[..., 1:] - theta_q[..., :-1]) / NSC[:, None, None]).clamp(min=0.0)

    # ------------------------------------------------------------ telemetry
    #: Per-candidate tables get parked in the npz; the jsonl keeps only what a
    #: human greps. Keys listed here are moved out of the JSON record.
    _ARRAY_KEYS = ("J_full", "J_trunc", "clip_frac", "nu_mean",
                   "breakdown_mean", "valid_steps", "winner_action",
                   "probe_lag_mean", "probe_lag_end", "probe_dev_mean",
                   "probe_theta_adv_hat", "probe_theta_adv_asgn")

    def _log_decision(self) -> None:
        dec = getattr(self._cem, "last_decision", None)
        if dec is None or self._log_fp is None:
            return
        rec = {"replan_idx": self._replan_idx, "env_step": self._env_step,
               "mode": self._mode, "library": self._families}
        for k, v in dec.items():
            if k in self._ARRAY_KEYS:
                self._push(k, v)
            elif k == "post" and isinstance(v, dict):
                for kk in ("refined_action", "seed_action"):
                    if kk in v:
                        self._push(kk, v.pop(kk))
                rec["post"] = v
            elif k == "meta":
                if "meta" not in self._arrays:
                    self._arrays["meta"] = [json.dumps(v)]     # constant across replans
            else:
                rec[k] = v
        self._push("env_step", self._env_step)
        # post-CEM block is small (per-env scalars) — keep it in the jsonl where
        # it is greppable, since it is what answers "why was the seed dropped?".
        if getattr(self, "_last_sup", None):
            rec.update(self._last_sup); self._last_sup = None
        self._log_fp.write(json.dumps(rec) + "\n")
        self._log_fp.flush()
        if self._replan_idx % 20 == 0:
            self._flush_arrays()
        self._cem.last_decision = None

    def _push(self, key: str, val: Any) -> None:
        if isinstance(val, dict):
            for kk, vv in val.items():
                self._push(f"{key}__{kk}", vv)
            return
        self._arrays.setdefault(key, []).append(np.asarray(val, dtype=np.float32))

    def _flush_arrays(self) -> None:
        """Stack and write ``cand_arrays.npz`` (overwritten; cheap at this size)."""
        if not self._arrays or self._outdir is None:
            return
        out = {}
        for k, v in self._arrays.items():
            if k == "meta":
                out[k] = np.asarray(v)
                continue
            try:
                out[k] = np.stack(v)
            except ValueError:
                continue        # ragged (shouldn't happen) — skip rather than crash
        np.savez_compressed(self._outdir / "cand_arrays.npz", **out)

    def close(self) -> None:
        self._flush_arrays()
        if self._log_fp is not None:
            self._log_fp.close()
            self._log_fp = None
