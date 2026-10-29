"""EnsembleMPCCSolver — diffusion-MPCC ONLY. GT-MPCC is unaffected (uses
`MultiviewMPCCSolver` via `mpcc_multiview.yaml`).

Mirrors `latent_mpc.solver_ensemble.EnsembleMPCSolver`:
  - Subclass `MultiviewMPCCSolver`. Base classes' bytecode untouched.
  - At each replan, the diffusion policy samples N candidate ReferenceTrajectory
    per env; this solver runs a SINGLE batched CEM with effective batch
    B = n_envs * N. Each (env, candidate) row gets its own ν-augmented
    action plan. After CEM, per-env argmin picks the candidate the WM
    "agrees with" (lowest contour cost).

MPCC-specific differences vs MPC ensemble:
  - Inner CEM solves on augmented action space (A_phys + 1) — ν is the last
    slot. Init_action ν-padding mirrors `MPCCSolver.solve` (phys-only init
    gets padded with `nu_init` or `_gt_nu_value`).
  - **Output ν-strip**: returned actions are `(n_envs, H, A_phys)` —
    WorldModelPolicy only sees physical actions, never ν. Mirrors
    `MPCCSolver.solve:277`.

Why safe vs GT-MPCC:
  - New file, new class. Imported only by Hydra when
    `cfg.solver=mpcc_multiview_ensemble`.
  - n_ensemble=1 short-circuits to `super().solve()` — bit-perfect parity
    with non-ensemble path (just goes through MultiviewMPCCSolver →
    MPCCSolver, which already does ν-strip).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from eval_mpcc_multiview import MultiviewMPCCSolver


class EnsembleMPCCSolver(MultiviewMPCCSolver):
    """Multiview MPCC solver with N-ensemble references per env, batched."""

    def __init__(self, *args, n_ensemble: int = 1, verbose: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_ensemble = int(n_ensemble)
        if self._n_ensemble < 1:
            raise ValueError(f"n_ensemble must be >= 1, got {n_ensemble}")
        # Filled in `configure`. Used to contract output back to real shape.
        self._real_n_envs: int | None = None
        # Per-solve verbose printing of N-candidate cost matrix + CEM convergence
        self._verbose = bool(verbose)
        self._n_solves = 0

    # ------------------------------------------------------------------
    def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
        """Inflate effective n_envs by N. CEM allocates state (mean, std,
        action_clamp tile) for n_envs * N independent rows. cost_model's
        ``self.references`` (nn.ModuleList) is sized n_envs * N — caller must
        populate ``solver._references`` (list of ReferenceTrajectory) and
        ``solver._env_ref_indices`` (np.arange(n_envs*N)) BEFORE
        ``world.set_policy(policy)``."""
        self._real_n_envs = int(n_envs)
        super().configure(
            action_space=action_space,
            n_envs=int(n_envs) * self._n_ensemble,
            config=config,
        )

    # ------------------------------------------------------------------
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        """One batched CEM solve across n_envs * N independent rows.

        Pre-encodes pixels ONCE at real-n batch (saves N× encoder work),
        ν-pads init_action if phys-only, expands ``info_dict`` and the cached
        emb to the inflated batch, delegates to the inner CEM. After CEM
        returns, re-evaluates per-row cost on the elite mean and picks the
        per-env best candidate via argmin over N. Finally strips ν before
        returning to policy.
        """
        N = self._n_ensemble
        if N == 1:
            # Bit-perfect parity with MultiviewMPCCSolver (which already
            # handles ν-pad / ν-strip / GT-init via super chain).
            return super().solve(info_dict, init_action=init_action)

        real_n = self._real_n_envs
        device = self._device

        # --- GT-init pop (mirrors MPCCSolver.solve:202-208) ---
        if (init_action is None
                and self._gt_init_actions is not None
                and self._gt_solve_count < self._gt_init_actions.shape[0]):
            init_action = self._gt_init_actions[self._gt_solve_count].to(device)
            self._gt_solve_count += 1

        # --- Inject per-row ref idx on device (sized real_n * N) ---
        # MPCC has no _env_ref_steps (advance_steps is no-op); only _ref_idx
        # routes envs to their ReferenceTrajectory in cost_model.references.
        ref_idx_dev = torch.as_tensor(
            self._env_ref_indices, dtype=torch.long, device=device
        )
        info_dict["_ref_idx"] = ref_idx_dev

        # --- Pre-encode at real_n batch (cheap; N× saved) ---
        with torch.no_grad():
            encode_input: dict[str, Any] = {
                "pixels": info_dict["pixels"].to(device, non_blocking=True),
            }
            for _key in ("wrist_pixels", "proprio_gripper_opening",
                         "proprio_effector_pos"):
                if _key in info_dict and torch.is_tensor(info_dict[_key]):
                    encode_input[_key] = info_dict[_key].to(device, non_blocking=True)
            self._model.encode(encode_input)
            emb_real = encode_input["emb"]   # (real_n, T_hist, D)

        # --- Build expanded info_dict (batch dim real_n → real_n * N) ---
        # _ref_idx is already (real_n*N,) so the shape check
        # ``v.shape[0] == real_n`` is False — passes through unchanged.
        expanded: dict = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == real_n:
                expanded[k] = v.repeat_interleave(N, dim=0)
            else:
                expanded[k] = v
        # Cache the encoded emb on expanded (skips redundant encode inside CEM
        # cost path; jepa rollout's use_precomputed_emb branch picks it up).
        expanded["emb"] = emb_real.repeat_interleave(N, dim=0)

        # --- Augment init_action with ν if phys-only (mirrors MPCCSolver.solve:239-259) ---
        if init_action is not None:
            A_in = init_action.shape[-1]
            if A_in == self._physical_action_dim:
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
                    f"init_action last dim {A_in} ≠ phys "
                    f"({self._physical_action_dim}) or phys+1 "
                    f"({self._physical_action_dim + 1})"
                )

            # Expand init_action to (real_n * N, ...) if currently real_n.
            if torch.is_tensor(init_action) and init_action.shape[0] == real_n:
                init_action = init_action.repeat_interleave(N, dim=0)

        # --- Inner CEM (batch_size in yaml should be ≥ real_n * N for one-shot) ---
        inner_outputs = self._inner_solver.solve(expanded, init_action=init_action)
        actions_aug = inner_outputs["actions"]   # (real_n * N, H, A_phys + 1)

        # --- Per-row cost re-evaluation on elite mean -------------------
        with torch.no_grad():
            cost_info: dict = {}
            for k, v in expanded.items():
                if k == "_ref_idx":
                    cost_info[k] = v   # already on device, keep rank-1
                elif torch.is_tensor(v):
                    v_dev = v.to(device, non_blocking=True)
                    if v_dev.dim() >= 1 and v_dev.shape[0] == real_n * N:
                        cost_info[k] = v_dev.unsqueeze(1)   # (B, 1, ...)
                    else:
                        cost_info[k] = v_dev
                else:
                    cost_info[k] = v
            actions_for_cost = actions_aug.unsqueeze(1).to(device, non_blocking=True)   # (B, 1, H, A_phys+1)
            cost = self.cost_model.get_cost(cost_info, actions_for_cost)   # (B, 1)
            if cost.dim() >= 2:
                cost = cost.squeeze(1)
            cost = cost.detach()

        # --- Reshape and pick per-env best candidate --------------------
        # actions_aug shape: (real_n * N, H, A_phys + 1)
        actions_g = actions_aug.view(real_n, N, *actions_aug.shape[1:])
        cost_g = cost.view(real_n, N)
        best_n = cost_g.argmin(dim=1).to(actions_aug.device)                 # (real_n,)
        chosen_aug = actions_g[
            torch.arange(real_n, device=actions_aug.device), best_n
        ]   # (real_n, H, A_phys + 1)

        # --- Verbose diagnostics: per-env cost matrix + CEM convergence ---
        if self._verbose:
            self._n_solves += 1
            costs_np = cost_g.detach().cpu().numpy()                     # (real_n, N)
            best_np = best_n.detach().cpu().numpy() if torch.is_tensor(best_n) else best_n
            print(f"\n[mpcc ensemble solve #{self._n_solves}] N={N}, n_envs={real_n}")
            print("  post-CEM cost (each row = env, *=picked):")
            for env_i in range(real_n):
                cells = "  ".join(
                    f"{costs_np[env_i, e]:7.4f}{'*' if e == int(best_np[env_i]) else ' '}"
                    for e in range(N)
                )
                picked = int(best_np[env_i])
                print(f"    env {env_i}: [{cells}]  → picked N{picked} cost={costs_np[env_i, picked]:.4f}")
            ich = inner_outputs.get("iter_cost_history")
            if ich is not None and len(ich) >= real_n * N:
                init_v = np.asarray([ich[r][0] for r in range(real_n * N)])
                final_v = np.asarray([ich[r][-1] for r in range(real_n * N)])
                init_g = init_v.reshape(real_n, N)
                final_g = final_v.reshape(real_n, N)
                init_best = init_g[np.arange(real_n), best_np]
                final_best = final_g[np.arange(real_n), best_np]
                print(f"  CEM iter cost (picked candidate): "
                      f"initial→final per env: " +
                      " | ".join(f"e{i}: {init_best[i]:.3f}→{final_best[i]:.3f}"
                                 for i in range(real_n)))
                print(f"  CEM all-candidate range: "
                      f"initial mean={init_v.mean():.3f} (min={init_v.min():.3f}, max={init_v.max():.3f}) → "
                      f"final mean={final_v.mean():.3f} (min={final_v.min():.3f}, max={final_v.max():.3f})")

        # --- Per-solve log (mirrors MPCCSolver.solve:265-274) — capture
        # the chosen augmented action BEFORE ν strip so downstream analysis
        # sees the full plan including ν.
        self._solve_log.append({
            "actions":    chosen_aug.clone(),  # (real_n, H, A_phys+1)
            "costs":      list(inner_outputs.get("costs", [])),
            "init_l2":    list(inner_outputs.get("init_l2", [])),
            "final_l2":   list(inner_outputs.get("final_l2", [])),
            "init_aabs":  list(inner_outputs.get("init_aabs", [])),
            "final_aabs": list(inner_outputs.get("final_aabs", [])),
            "init_nu":    list(inner_outputs.get("init_nu", [])),
            "final_nu":   list(inner_outputs.get("final_nu", [])),
            "best_n":     best_n.detach().cpu().numpy() if torch.is_tensor(best_n) else best_n,
            "all_costs":  cost_g.detach().cpu().numpy(),
        })

        # --- ν strip — WorldModelPolicy only sees physical actions ---
        chosen_phys = chosen_aug[..., :self._physical_action_dim]   # (real_n, H, A_phys)

        return {
            "actions": chosen_phys,
            "best_n": best_n.detach().cpu().numpy() if torch.is_tensor(best_n) else best_n,
            "all_costs": cost_g.detach().cpu().numpy(),
            "iter_cost_history": inner_outputs.get("iter_cost_history"),
        }
