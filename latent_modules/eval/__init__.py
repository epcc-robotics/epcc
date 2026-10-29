"""Shared eval-time logging utilities.

Produces task-agnostic rollout dumps that the plotting pipeline (Section 7 of
the CoRL'26 paper) consumes uniformly across all method × task cells.

Layout per run:
    <results_path>/
        rollout.npz       per-episode + per-step arrays
        meta.yaml         method / task / hparam / ckpt provenance
        keyframes/        ep_{i}_{k}.png, k in 0..N_keyframes-1
"""

from latent_modules.eval.rollout_logger import RolloutLogger
from latent_modules.eval.success_specs import (
    SUCCESS_SPECS,
    SuccessSpec,
    get_success_spec,
)

__all__ = [
    "RolloutLogger",
    "SUCCESS_SPECS",
    "SuccessSpec",
    "get_success_spec",
    "compute_demo_ttc_for_run",
]


def compute_demo_ttc_for_run(*args, **kwargs):
    """Lazy re-export to avoid importing swm at top-level."""
    from latent_modules.eval.demo_ttc import compute_demo_ttc_for_run as _impl
    return _impl(*args, **kwargs)
