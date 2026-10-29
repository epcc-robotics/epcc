"""EPCC-5 candidate-library DTG-MPCC (SELECT / seeded rows) with the warm history ring.

``DTGMPCCCandidatePolicy`` inherits the first-replan history-ring defect described in
policy/dtg_mpcc_warm.py — its ``super().reset`` builds the same inner
``DiffusionMPCCPolicy`` whose ``_update_ring`` duplicates the current encoding on the
first call. This subclass applies the same seeding as ``DTGMPCCWarmPolicy`` after the
candidate machinery is set up, so the Table II SELECT / EPCC(seeded) rows can be re-run
under the same DTG-side fix the EPCC row uses (2026-09-07 campaign).

Scope: the DTG latent ring only. The DP candidate library's pixel ring (``_dp_ring`` in
policy/dtg_mpcc_cand.py) still cold-starts by duplicating the current frame — deliberately,
because that is exactly how the DP / SuP baseline rows of the same table run
(eval_cleandiffuser_dp.py), so the library the world model scores is the one the DP row
executes. State this in the Table II caption.

Config: ``- override /policy: dtg_mpcc_cand_warm`` (or ``policy=dtg_mpcc_cand_warm`` on the
command line) plus the same ``warm_ring`` block the warm cells carry. Everything else —
``candidates.*``, ``solver.inner_solver._target_=...CandidateCEMSolver`` — is unchanged.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from policy.dtg_mpcc_cand import DTGMPCCCandidatePolicy
from policy.dtg_mpcc_warm import seed_warm_ring


class DTGMPCCCandidateWarmPolicy(DTGMPCCCandidatePolicy):
    """DTGMPCCCandidatePolicy whose first plan is conditioned on real history."""

    def reset(self, world: Any, cfg: DictConfig | None = None,
              eval_world: Any | None = None) -> None:
        super().reset(world, cfg=cfg, eval_world=eval_world)   # DTG + candidate setup
        seed_warm_ring(self, cfg, eval_world)                  # before the first get_action
