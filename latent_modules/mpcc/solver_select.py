"""EPCC-5 candidate selector — a CEM-free inner solver for EPCC-SELECT.

Design note
-----------
This subclasses ``stable_worldmodel.solver.CEMSolver`` rather than
``MPCCSolver`` so that the **entire** MPCC preamble is reused untouched:
per-env reference injection, one-shot pixel encoding, ν-augmentation of the
warm start, the per-solve log, and the final ν strip all live in
``MPCCSolver.solve`` and are agnostic to which inner solver runs.  Swapping is
a pure config change::

    solver.inner_solver._target_: latent_modules.mpcc.solver_select.CandidateCEMSolver

``swm_srl_v2`` is shared with other projects and is never edited.

Contract
--------
The policy calls :meth:`set_candidates` immediately before the replan.  With
candidates present ``solve`` performs **one** batched world-model rollout over
all M candidates and returns the arg-min; with none it falls straight through
to the stock CEM, so the same class can serve non-SELECT rows.

Two scoring rules (EPCC-5 plan §1.4), both computed from the *same* rollout:

``J_full``
    Full H-step horizon.  Identical objective to REFINE / Full EPCC.  The
    padded tail (constant-velocity extrapolation past the DP chunk) is scored,
    which penalises high rates ``k`` — i.e. it *handicaps* SELECT.
``J_trunc``
    Steps past the candidate's real reach are masked out and the remaining sum
    is normalised by the number of valid steps.  Bias runs the other way: high
    ``k`` is judged on a shorter, easier look-ahead.

Reporting both brackets the padding confound.  ``last_decision`` carries the
per-replan telemetry (both arg-mins and their disagreement) for
``cand_decisions.jsonl``.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from stable_worldmodel.solver import CEMSolver


def _best_per_group(j: torch.Tensor, key: list) -> dict:
    """Min cost per group (rate / family), averaged over envs."""
    out: dict = {}
    for g in sorted(set(key)):
        cols = [i for i, k in enumerate(key) if k == g]
        out[str(g)] = float(j[:, cols].min(dim=1).values.mean())
    return out


def _rank_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean Spearman correlation between two (E, M) score tables, over envs."""
    if a.shape[1] < 3:
        return float("nan")
    ra = a.argsort(1).argsort(1).float()
    rb = b.argsort(1).argsort(1).float()
    ra = ra - ra.mean(1, keepdim=True)
    rb = rb - rb.mean(1, keepdim=True)
    num = (ra * rb).sum(1)
    den = ra.norm(dim=1) * rb.norm(dim=1) + 1e-12
    return float((num / den).mean())


class CandidateCEMSolver(CEMSolver):
    """Evaluate a fixed candidate library and execute the lowest-cost member."""

    #: Set by the policy: ``builder(info_dict, solver) -> (cand, valid, meta)``.
    #: Invoked at solve time, i.e. AFTER the DTG contour for this replan has been
    #: pushed into the cost model, so ``S^(m)`` reads the current ``ref.theta``.
    candidate_builder = None

    def __init__(self, *args: Any, scoring: str = "full", mode: str = "select",
                 probe: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if scoring not in ("full", "trunc"):
            raise ValueError(f"scoring must be 'full' or 'trunc', got {scoring!r}")
        if mode not in ("select", "refine", "full_cand", "refine_sup"):
            raise ValueError(
                f"mode must be select|refine|full_cand|refine_sup, got {mode!r}")
        self.scoring = scoring
        self.mode = mode
        #: G1 alignment probe. Adds one contour projection per candidate, so it is
        #: opt-in (`++solver.inner_solver.probe=true`) and used on the gate cells,
        #: not the 22-cell grid.
        self.probe = bool(probe)
        self._cand: torch.Tensor | None = None       # (n_envs, M, H, A_aug)
        self._cand_valid: torch.Tensor | None = None  # (n_envs, M) real CEM steps
        self._cand_meta: list[dict] | None = None     # M entries: {family, k, pad_frac}
        # refine_sup: the seed is chosen by SuP's own ADM/IQL gate, not by our cost.
        # Kept separate from the arg-min so both are computed and logged every replan --
        # "did SuP agree with us" is the whole point of the composability row.
        self._seed_idx: torch.Tensor | None = None
        self.last_decision: dict | None = None

    # ------------------------------------------------------------------ API
    def set_candidates(
        self,
        candidates: torch.Tensor,
        valid_steps: torch.Tensor | np.ndarray,
        meta: list[dict] | None = None,
        seed_idx: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        """Stage the library for the next :meth:`solve`.

        Args:
            candidates: ``(n_envs, M, H, A_phys+1)`` in **normalized** action
                space, ν already in the last column.
            valid_steps: ``(n_envs, M)`` or ``(M,)`` — number of leading CEM
                steps backed by real policy output (the rest is padding).
                Clamped into ``[1, horizon]``.
            meta: optional per-candidate descriptors, logged verbatim.
            seed_idx: ``(n_envs,)`` candidate index to seed the CEM from, chosen
                OUTSIDE this solver (mode="refine_sup"). ``None`` keeps the cost
                arg-min, which is what every other mode uses.
        """
        if candidates.ndim != 4:
            raise ValueError(f"candidates must be (n_envs, M, H, A_aug); got {tuple(candidates.shape)}")
        n_envs, M, H, A = candidates.shape
        if H != self.horizon:
            raise ValueError(f"candidate horizon {H} != solver horizon {self.horizon}")
        if self._nu_slice is None:
            raise ValueError("CandidateCEMSolver requires MPCC mode (physical_dim set)")
        if A != self.action_dim:
            raise ValueError(f"candidate action dim {A} != solver action_dim {self.action_dim}")

        v = torch.as_tensor(valid_steps)
        if v.ndim == 1:
            v = v[None].expand(n_envs, M)
        # NOT clamp_: `v` may be a stride-0 broadcast view (expand), which in-place
        # ops reject. Out-of-place clamp materialises a dense tensor.
        v = v.to(torch.long).clamp(1, H).contiguous()

        self._cand = candidates.to(self.device)
        self._cand_valid = v.to(self.device)
        self._cand_meta = meta
        if seed_idx is None:
            self._seed_idx = None
        else:
            si = torch.as_tensor(seed_idx).reshape(-1).to(torch.long)
            if si.numel() != n_envs:
                raise ValueError(f"seed_idx has {si.numel()} entries, expected {n_envs}")
            self._seed_idx = si.clamp(0, M - 1).to(self.device)

    def clear_candidates(self) -> None:
        self._cand = None
        self._cand_valid = None
        self._cand_meta = None
        self._seed_idx = None

    # ------------------------------------------------------------- internals
    def _shape_candidates(self, cand: torch.Tensor) -> torch.Tensor:
        """Apply the exact shaping CEM applies to its own samples.

        Mirrors ``cem.py`` MPCC branch: norm-space ``action_clamp`` on the
        physical slice → env-space projection through ``env_clip_range`` →
        ``nu_clamp`` on ν.  Without this, SELECT would search a different
        feasible set than REFINE / Full and the ladder would not be matched.
        """
        cand = cand.clone()
        phys = cand[..., self._phys_slice]
        if self._clamp_tensor is not None:
            phys = torch.clamp(phys, min=-self._clamp_tensor, max=self._clamp_tensor)
        elif isinstance(self.action_clamp, (int, float)):
            phys = phys.clamp(-self.action_clamp, self.action_clamp)
        if (self.projection_mode != "none"
                and self._scaler_mean_tiled is not None
                and self.env_clip_range is not None):
            phys = self._project_env(phys)
        cand[..., self._phys_slice] = phys
        if self.nu_clamp is not None:
            cand[..., self._nu_slice] = cand[..., self._nu_slice].clamp(*self.nu_clamp)
        return cand

    def _to_env(self, phys: torch.Tensor) -> torch.Tensor:
        """Normalized → env action units, so logged plans line up with rollouts."""
        if self._scaler_mean_tiled is None:
            return phys.detach().cpu()
        return (phys * self._scaler_scale_tiled + self._scaler_mean_tiled).detach().cpu()

    def _clip_frac(self, cand: torch.Tensor) -> torch.Tensor:
        """Per-candidate fraction of physical entries the boxes would bite.

        A candidate that loses because ``action_clamp`` / ``env_clip_range``
        truncated it is a different story from one the world model disliked;
        this column keeps the two apart.
        """
        phys = cand[..., self._phys_slice]
        hit = torch.zeros(phys.shape[:2], device=phys.device)
        n = float(phys[0, 0].numel())
        if self._clamp_tensor is not None:
            hit = hit + (phys.abs() >= self._clamp_tensor * (1 - 1e-6)).float().sum((-1, -2)) / n
        elif self._scaler_mean_tiled is not None and self.env_clip_range is not None:
            env = phys * self._scaler_scale_tiled + self._scaler_mean_tiled
            lo, hi = self.env_clip_range
            hit = hit + ((env <= float(lo) + 1e-6) | (env >= float(hi) - 1e-6)).float().sum((-1, -2)) / n
        return hit.detach().cpu()

    def _trunc_cost(self, per_step: dict, valid: torch.Tensor) -> torch.Tensor:
        """J_trunc: mask padded steps, normalise by the valid-step count."""
        m = self.model
        if float(getattr(m, "q_lag", 0.0)) != 0.0:
            raise NotImplementedError(
                "scoring='trunc' assumes q_lag=0 (contour_mode=legacy). "
                "Per-step lag is not stashed."
            )
        contour_t = per_step["contour_t"]          # (B, M, T)  T = H+1
        nu = per_step["nu"]                        # (B, M, H)
        action_t = per_step["action_t"]            # (B, M, H)
        physical = per_step["physical"]            # (B, M, H, A_phys)
        B, M, H = nu.shape
        dev = nu.device
        n = valid.to(dev).to(nu.dtype)             # (B, M)

        step = torch.arange(H, device=dev)[None, None, :]        # j = 0..H-1
        mask_h = (step < valid[..., None]).to(nu.dtype)          # (B, M, H)
        # contour index t=0 is the CURRENT latent (always valid); t=j+1 pairs with step j.
        mask_t = torch.cat([torch.ones(B, M, 1, device=dev, dtype=nu.dtype), mask_h], dim=-1)

        contour = (contour_t * mask_t).sum(-1)
        progress = (nu * mask_h).sum(-1)
        action = (action_t * mask_h).sum(-1)

        smooth = torch.zeros_like(progress)
        q_s = float(getattr(m, "q_action_smooth", 0.0))
        if q_s > 0.0:
            ab = int(m.action_block)
            pd = physical.shape[-1] // ab
            phys_r = physical.reshape(B, M, H, ab, pd)
            if getattr(m, "action_smooth_all_dims", False):
                seq = phys_r.reshape(B, M, H * ab, pd)
                diff = seq[..., 1:, :] - seq[..., :-1, :]        # (B, M, H·ab-1, pd)
                per = diff.pow(2).sum(-1)                        # (B, M, H·ab-1)
            else:
                seq = phys_r[..., -1].reshape(B, M, H * ab)
                per = (seq[..., 1:] - seq[..., :-1]).pow(2)      # (B, M, H·ab-1)
            # diff i spans env-actions i and i+1 → valid iff i+1 < n·ab
            idx = torch.arange(H * ab - 1, device=dev)[None, None, :]
            dmask = ((idx + 1) < (valid[..., None] * ab)).to(nu.dtype)
            smooth = (per * dmask).sum(-1)

        total = (float(m.q_contour) * contour
                 - float(m.q_progress) * progress
                 + float(m.q_action) * action
                 + q_s * smooth)
        return total / n.clamp(min=1.0)

    def _score_plan(self, info_dict: dict, plan: torch.Tensor):
        """Cost of a single (n_envs, H, A_aug) plan under the identical objective.

        One extra world-model rollout per replan — negligible next to the CEM's
        ``num_samples × n_steps`` — and it is what lets the log answer "was the
        refined action actually better, and in which term?" without a re-run.
        """
        n_envs = plan.shape[0]
        js, bds = [], []
        for start in range(0, n_envs, self.batch_size):
            end = min(start + self.batch_size, n_envs)
            bs = end - start
            exp: dict[str, Any] = {}
            for k, v in info_dict.items():
                vb = v[start:end]
                if torch.is_tensor(vb):
                    vb = vb.to(self.device, non_blocking=True).unsqueeze(1).expand(bs, 1, *vb.shape[1:])
                elif isinstance(vb, np.ndarray):
                    vb = np.repeat(vb[:, None, ...], 1, axis=1)
                exp[k] = vb
            j = self.model.get_cost(exp, plan[start:end].unsqueeze(1).to(self.device))
            js.append(j.detach().cpu()[:, 0])
            bd = {kk: vv.detach().cpu()[:, 0]
                  for kk, vv in (getattr(self.model, "_last_breakdown", None) or {}).items()
                  if torch.is_tensor(vv) and vv.ndim == 2}
            bds.append(bd)
        J = torch.cat(js, 0)
        BD = {kk: torch.cat([b[kk] for b in bds], 0) for kk in bds[0]} if bds else {}
        return J, BD

    # ----------------------------------------------------------------- solve
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        # Build the library for THIS replan. Deferred to solve time on purpose:
        # DiffusionMPCCPolicy refreshes the contour just before the solver runs,
        # so only here does ``cost_model.references`` hold the current ref.
        if self._cand is None and self.candidate_builder is not None:
            built = self.candidate_builder(info_dict, self)
            if built is not None:
                self.set_candidates(*built)
        if self._cand is None:
            return super().solve(info_dict, init_action=init_action)

        t0 = time.time()
        cand = self._shape_candidates(self._cand)
        n_envs, M, H, A = cand.shape
        chosen = torch.empty(n_envs, H, A, device=self.device)
        j_full_all: list[torch.Tensor] = []
        j_trunc_all: list[torch.Tensor] = []
        bd_all: dict[str, list[torch.Tensor]] = {}

        prev_flag = getattr(self.model, "emit_per_step", False)
        prev_probe = getattr(self.model, "emit_probe", False)
        self.model.emit_per_step = True
        self.model.emit_probe = self.probe
        probe_all: dict[str, list[torch.Tensor]] = {}
        try:
            for start in range(0, n_envs, self.batch_size):
                end = min(start + self.batch_size, n_envs)
                bs = end - start

                expanded: dict[str, Any] = {}
                for k, v in info_dict.items():
                    vb = v[start:end]
                    if torch.is_tensor(vb):
                        vb = vb.to(self.device, non_blocking=True)
                        vb = vb.unsqueeze(1).expand(bs, M, *vb.shape[1:])
                    elif isinstance(vb, np.ndarray):
                        vb = np.repeat(vb[:, None, ...], M, axis=1)
                    expanded[k] = vb

                sub = cand[start:end]                              # (bs, M, H, A)
                j_full = self.model.get_cost(expanded, sub)        # (bs, M)
                per_step = self.model._last_per_step
                j_trunc = self._trunc_cost(per_step, self._cand_valid[start:end])

                score = j_full if self.scoring == "full" else j_trunc
                m_star = score.argmin(dim=1)                       # (bs,)
                if self._seed_idx is not None:
                    # SuP picked the rate; our cost still scores everything so the
                    # arg-min stays in the log next to what SuP asked for.
                    m_star = self._seed_idx[start:end]
                chosen[start:end] = sub[torch.arange(bs, device=self.device), m_star]
                j_full_all.append(j_full.detach().cpu())
                j_trunc_all.append(j_trunc.detach().cpu())
                # Per-term breakdown (contour / progress / action / smooth) so the
                # probe can ask what actually drives the arg-min — e.g. whether a
                # candidate lost on contour error or simply on assigned pace.
                for kk, vv in (getattr(self.model, "_last_breakdown", None) or {}).items():
                    if torch.is_tensor(vv) and vv.shape[:2] == j_full.shape[:2]:
                        bd_all.setdefault(kk, []).append(vv.detach().cpu())
                if self.probe:
                    pr = getattr(self.model, "_last_probe", None) or {}
                    if pr:
                        # t=0 is the current latent: it projects onto θ₀ by
                        # construction, so its lag is trivially 0. Score t≥1 only.
                        lag = (pr["theta_hat"] - pr["theta_asgn"])[..., 1:]
                        probe_all.setdefault("lag_mean", []).append(lag.mean(-1).detach().cpu())
                        probe_all.setdefault("lag_end", []).append(lag[..., -1].detach().cpu())
                        probe_all.setdefault("dev_mean", []).append(
                            pr["dist"][..., 1:].mean(-1).detach().cpu())
                        probe_all.setdefault("theta_adv_hat", []).append(
                            (pr["theta_hat"][..., -1] - pr["theta_hat"][..., 0]).detach().cpu())
                        probe_all.setdefault("theta_adv_asgn", []).append(
                            (pr["theta_asgn"][..., -1] - pr["theta_asgn"][..., 0]).detach().cpu())
        finally:
            self.model.emit_per_step = prev_flag
            self.model.emit_probe = prev_probe

        jf = torch.cat(j_full_all, 0)                              # (n_envs, M)
        jt = torch.cat(j_trunc_all, 0)
        clip = self._clip_frac(self._cand)                         # (n_envs, M)
        nu_mean = self._cand[..., self._nu_slice].squeeze(-1).mean(-1).detach().cpu()
        rate_of = [float(m.get("k", float("nan"))) for m in (self._cand_meta or [{}] * M)]
        fam_of = [str(m.get("family", "?")) for m in (self._cand_meta or [{}] * M)]
        # The action box must ADMIT the library, for the same reason the ν cap
        # must: a truncated accumulate-k chunk is no longer "rate k", yet its
        # schedule still asks for k waypoints per step, so the candidate loses on
        # a mismatch we introduced rather than on the operator's merit. Shout
        # once if that is happening — it must not pass silently in a 22-cell run.
        if not getattr(self, "_clip_warned", False):
            worst = float(clip.max())
            if worst > 0.01:
                by_rate = {
                    g: round(float(clip[:, [i for i, k in enumerate(rate_of) if k == g]].max()), 3)
                    for g in sorted(set(rate_of))
                }
                logging.warning(
                    "[SELECT] action box is biting the library: max clip_frac=%.3f "
                    "(per-rate max: %s). Raise wall (and env_clip_range) so every "
                    "rate is admissible, or the high-k candidates are handicapped.",
                    worst, by_rate,
                )
                self._clip_warned = True
        arg_f = jf.argmin(1)
        arg_t = jt.argmin(1)
        self.last_decision = {
            "scoring": self.scoring,
            "M": int(M),
            "argmin_full": arg_f.tolist(),
            "argmin_trunc": arg_t.tolist(),
            "argmin_disagree": float((arg_f != arg_t).float().mean()),
            "seed_idx": (self._seed_idx.tolist() if self._seed_idx is not None else None),
            "seed_matches_argmin": (
                None if self._seed_idx is None
                else float((self._seed_idx.cpu() == (arg_f if self.scoring == "full" else arg_t))
                           .float().mean())),
            "J_full": jf.tolist(),
            "J_trunc": jt.tolist(),
            "valid_steps": self._cand_valid.detach().cpu().tolist(),
            "meta": self._cand_meta,
            "solve_ms": (time.time() - t0) * 1e3,
            "breakdown_mean": {
                kk: torch.cat(vv, 0).mean(0).tolist() for kk, vv in bd_all.items()
            },
            # Which candidate each single term would have picked on its own —
            # separates "won on pace" from "won on contour accuracy".
            "argmin_per_term": {
                kk: torch.cat(vv, 0).argmin(1).tolist() for kk, vv in bd_all.items()
            },
            # Spearman-style agreement between the two scoring rules across the
            # whole library (stronger than the binary arg-min disagreement).
            "rank_corr_full_trunc": _rank_corr(jf, jt),
            "clip_frac": clip.tolist(),
            "nu_mean": nu_mean.tolist(),
            # Index-level disagreement over-counts: the 8 DDPM draws that share a
            # rate are interchangeable, so two rules can pick the SAME rate and
            # still differ by index. These are the metrics to actually report.
            "argmin_full_rate": [rate_of[i] for i in arg_f.tolist()],
            "argmin_trunc_rate": [rate_of[i] for i in arg_t.tolist()],
            "rate_disagree": float(np.mean([
                rate_of[i] != rate_of[j] for i, j in zip(arg_f.tolist(), arg_t.tolist())])),
            "family_disagree": float(np.mean([
                fam_of[i] != fam_of[j] for i, j in zip(arg_f.tolist(), arg_t.tolist())])),
            # Best achievable cost at each rate — the frontier the selector saw.
            "best_per_rate_full": _best_per_group(jf, rate_of),
            "best_per_rate_trunc": _best_per_group(jt, rate_of),
            # Winner in ENV units — directly comparable with rollout_traj.npz.
            "winner_action": self._to_env(chosen[..., self._phys_slice]).tolist(),
        }
        if probe_all:
            # G1: does the world-model rollout of each candidate actually land
            # where its schedule says?  lag = θ̂ − θ_assigned (negative = the chunk
            # falls behind the pace we assigned it); dev = ⊥ distance to the contour.
            self.last_decision.update(
                {f"probe_{k}": torch.cat(v, 0).tolist() for k, v in probe_all.items()})

        picked = arg_f if self.scoring == "full" else arg_t
        sel = jf if self.scoring == "full" else jt
        gather = sel[torch.arange(n_envs), picked]

        if self.mode != "select":
            # REFINE / Full(cand-init): the winner becomes the CEM initial mean.
            # ν stays pinned when the recipe sets nu_var_scale=0 (REFINE); with the
            # normal nu_var_scale it is free (Full(cand-init)).  The library and the
            # arg-min above are byte-identical to the SELECT row — the three rows
            # differ only in what is optimised afterwards.
            seed = chosen.clone()
            lib = cand                                   # keep for the "did it leave m*?" test
            self.clear_candidates()
            t_cem = time.time()
            # 🔴 CEMSolver.init_action_distrib does NOT copy init_action
            # (cem.py:348 `mean = ... else actions`) and the batch loop writes the
            # elite mean straight back into it (cem.py:657). Handing it `seed`
            # would overwrite our own reference copy in place, and every
            # seed-vs-refined diagnostic below would silently read 0.
            out = super().solve(info_dict, init_action=seed.clone())
            cem_ms = (time.time() - t_cem) * 1e3

            refined = out["actions"].to(self.device)     # (E, H, A_aug)
            j_ref, bd_ref = self._score_plan(info_dict, refined)
            j_seed, bd_seed = self._score_plan(info_dict, seed)

            d = (refined - seed)
            d_phys_norm = d[..., self._phys_slice].reshape(n_envs, -1).norm(dim=-1)
            sc = getattr(self, "_scaler_scale_tiled", None)
            d_phys_env = ((d[..., self._phys_slice] * sc) if sc is not None
                          else d[..., self._phys_slice]).reshape(n_envs, -1).norm(dim=-1)
            # Nearest library member to the refined plan: still m*, or effectively
            # a different rate?  This is the "did refinement leave the library" test.
            dist_to_lib = (refined.unsqueeze(1)[..., self._phys_slice]
                           - lib[..., self._phys_slice]).reshape(n_envs, M, -1).norm(dim=-1)
            nearest = dist_to_lib.argmin(1)

            self.last_decision.update({
                "post": {
                    "J_seed": j_seed.tolist(),
                    "J_refined": j_ref.tolist(),
                    "dJ": (j_ref - j_seed).tolist(),
                    "breakdown_seed": {kk: vv.tolist() for kk, vv in bd_seed.items()},
                    "breakdown_refined": {kk: vv.tolist() for kk, vv in bd_ref.items()},
                    "d_norm_space": d_phys_norm.detach().cpu().tolist(),
                    "d_env_space": d_phys_env.detach().cpu().tolist(),
                    "nearest_candidate": nearest.detach().cpu().tolist(),
                    "left_seed": (nearest != picked.to(nearest.device)).detach().cpu().tolist(),
                    "nu_seed": seed[..., self._nu_slice].squeeze(-1).detach().cpu().tolist(),
                    "nu_refined": refined[..., self._nu_slice].squeeze(-1).detach().cpu().tolist(),
                    "refined_action": self._to_env(refined[..., self._phys_slice]).tolist(),
                    "seed_action": self._to_env(seed[..., self._phys_slice]).tolist(),
                    "nu_pinned_ok": bool(torch.allclose(
                        seed[..., self._nu_slice], refined[..., self._nu_slice], atol=1e-6)),
                    "cem_ms": cem_ms,
                    "cem_iter_cost_first_last": [
                        [h[0], h[-1]] for h in out.get("iter_cost_history", []) if h
                    ],
                },
            })
            out["candidate_decision"] = self.last_decision
            return out
        nu_col = chosen[..., self._nu_slice].mean(dim=(1, 2)).detach().cpu()
        aabs = chosen[..., self._phys_slice].abs().mean(dim=(1, 2)).detach().cpu()

        actions = chosen.detach().cpu()
        nan = [float("nan")] * n_envs
        outputs = {
            "actions": actions,
            "costs": gather.tolist(),
            "mean": [actions],
            "var": [torch.zeros_like(actions)],
            "init_l2": nan,
            "final_l2": nan,
            "init_aabs": aabs.tolist(),
            "final_aabs": aabs.tolist(),
            "init_nu": nu_col.tolist(),
            "final_nu": nu_col.tolist(),
            "iter_cost_history": [[float(c)] for c in gather.tolist()],
        }
        self.clear_candidates()
        print(f"[SELECT] M={M} scoring={self.scoring} "
              f"rate_disagree={self.last_decision['rate_disagree']:.2f} "
              f"(idx {self.last_decision['argmin_disagree']:.2f}) "
              f"solve {(time.time() - t0):.3f}s")
        return outputs
