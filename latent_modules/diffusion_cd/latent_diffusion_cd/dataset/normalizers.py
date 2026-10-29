"""Normalizer factory — pluggable between StandardScaler / MinMax / Empty.

All implementations expose the same `(normalize, unnormalize)` API used by
CleanDiffuser dataset classes and downstream `.unnormalize()` at eval time.

API parity matches `cleandiffuser.dataset.dataset_utils.MinMaxNormalizer`
(see dataset_utils.py:338-365): constructor takes (N, ..., D) array, fits
per-last-dim stats. `normalize/unnormalize` accept any leading shape and
broadcast along the last dim.
"""
from __future__ import annotations

import numpy as np


class StandardScalerNormalizer:
    """Per-last-dim (x - mean) / std normalization. Match our existing
    latent_diffusion StandardScaler (used by the 70% SR baseline).

    For 384-d JEPA latents that are ~N(0,1) by construction, this is
    near-identity but still useful as a numerical safety net.

    Dead dims (std < 1e-6) get clamped to std=1.0 to avoid div-by-zero.
    """

    def __init__(self, X: np.ndarray):
        X = X.reshape(-1, X.shape[-1]).astype(np.float32)
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = X.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        return x * self.std + self.mean


def make_normalizer(kind: str, X: np.ndarray):
    """Factory: pick normalizer by string id.

    Args:
        kind: "standard" | "minmax" | "none"
        X: (N, ..., D) fit data

    Returns:
        instance with .normalize / .unnormalize methods
    """
    kind = (kind or "standard").lower()
    if kind == "standard":
        return StandardScalerNormalizer(X)
    if kind == "minmax":
        # Import here to avoid circular dependency on imagecodecs at module load.
        from cleandiffuser.dataset.dataset_utils import MinMaxNormalizer
        return MinMaxNormalizer(X)
    if kind == "none":
        from cleandiffuser.dataset.dataset_utils import EmptyNormalizer
        return EmptyNormalizer()
    raise ValueError(
        f"Unknown normalizer kind {kind!r}; expected 'standard' | 'minmax' | 'none'"
    )
