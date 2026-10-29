"""SpeedUpPatch-lite (SuP) primitives — action-chunk downsampling, per-env k
selection, and world-model latent-deviation scoring.

These are PURE helpers with no policy state, so the scoring function is
swappable and unit-testable independently of the DP wrapper in
``policy/sup_gated.py``.

Background (arXiv:2603.20658, SpeedUpPatch): a frozen base policy emits an
action chunk ``A`` of length T. An external scheduler picks a per-chunk
downsampling rate ``k`` and executes ``A^k = A[k-1::k]`` (equal-spacing
decimation), skipping intermediate waypoints so the agent advances ~k env
steps per control step. The full paper learns the scheduler with offline RL;
this SuP-lite chooses the largest ``k`` whose predicted state deviation stays
under ``eps`` (world-model gated), falling back to k=1 (no speedup) when unsure.

Convention: action chunks are ``(T, action_dim)`` for a single env, or
``(N_env, T, action_dim)`` batched. The time axis is always ``axis=-2``.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


# --------------------------------------------------------------------------- #
# Downsampling
# --------------------------------------------------------------------------- #
def downsample_actions(
    actions: np.ndarray, k: int, mode: str = "decimate"
) -> np.ndarray:
    """Downsample an action chunk along the time axis (``axis=-2``) by rate ``k``.

    Args:
        actions: ``(T, A)`` or ``(N, T, A)``. Position-like absolute actions
            (PushT: 2D target; no gripper).
        k: downsampling rate, ``>= 1``. ``k == 1`` is the identity (no copy of
            values — returned unchanged so k=1 stays byte-for-byte the base).
        mode: ``"decimate"`` keeps indices ``k-1, 2k-1, 3k-1, ...`` (the spec's
            equal-spacing decimation ``A[k-1::k]`` — phase-aligned so each kept
            action is the one that advances the env by k steps).

    Returns:
        The downsampled chunk, shorter along the time axis
        (``ceil((T-(k-1))/k)`` steps for decimate).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k == 1:
        return actions
    if mode == "decimate":
        # ABSOLUTE control (paper Eq.2 Abs): keep waypoints k-1, 2k-1, ...
        return actions[..., k - 1 :: k, :]
    if mode == "accumulate":
        # RELATIVE/Delta control (paper Eq.2 Delta): merge k consecutive deltas
        # by SUMMATION. A^k[j] = Σ a[jk : jk+k], length l = floor(T/k); the
        # remainder tail (< k) is dropped, matching the paper's (t : t+lk) span.
        T = actions.shape[-2]
        l = T // k
        if l == 0:
            raise ValueError(f"chunk length {T} < k={k}; cannot accumulate")
        head = actions[..., : l * k, :]  # drop remainder tail
        grouped = head.reshape(head.shape[:-2] + (l, k, head.shape[-1]))
        return grouped.sum(axis=-2)  # (..., l, A) — merged deltas
    raise ValueError(f"unknown downsample mode {mode!r}")


def gripper_aware_decimate(
    actions: np.ndarray, k: int, gripper_dim: int = -1
) -> np.ndarray:
    """Cube/manipulation variant: decimate the pose dims but never *skip* a
    gripper state change.

    Plain decimation can drop the single action where the gripper closes
    (grasp), turning a successful grasp into a miss. Here we decimate the pose
    channels normally, but OR-latch the gripper channel: the kept action at
    decimated index ``j`` takes the gripper value that is "most closed" across
    the ``k`` original actions it represents, so a close event inside the window
    is never lost.

    Args:
        actions: ``(T, A)`` or ``(N, T, A)``; the gripper is a single channel at
            ``gripper_dim`` (cube: last of 5). Assumes a convention where a
            larger gripper value ⇒ more closed (OR = ``max`` over the window).
        k: rate ``>= 1``.
        gripper_dim: index of the gripper channel.

    Returns:
        Decimated chunk with the gripper channel OR-latched over each window.

    NOTE: experimental — cube is a Phase-2 target. Validate the gripper polarity
    (max vs min = "closed") against the cube action convention before trusting
    grasp results.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k == 1:
        return actions
    T = actions.shape[-2]
    kept = downsample_actions(actions, k, mode="decimate")  # (..., T', A)
    # For each kept index j (original index k-1+j*k), latch the gripper over the
    # window [j*k, j*k + k) so an in-window close is preserved.
    grip = np.take(actions, indices=gripper_dim, axis=-1)  # (..., T)
    n_kept = kept.shape[-2]
    for j in range(n_kept):
        lo = j * k
        hi = min(lo + k, T)
        window_max = grip[..., lo:hi].max(axis=-1)  # (...,)
        kept[..., j, gripper_dim] = window_max
    return kept


# --------------------------------------------------------------------------- #
# k selection
# --------------------------------------------------------------------------- #
def choose_k_per_env(
    scores: np.ndarray, k_values: list[int], eps: float
) -> np.ndarray:
    """Pick, per env, the largest ``k`` whose deviation score is ``<= eps``.

    Args:
        scores: ``(N_env, K)`` deviation per env per candidate k, aligned to
            ``sorted(k_values)``. Score for k=1 must be 0 (identity ⇒ no
            deviation), so k=1 is always safe and the default.
        k_values: candidate rates, e.g. ``[1, 2, 3]``.
        eps: safety threshold in the score's units (WM latent-L2 for v0).

    Returns:
        ``(N_env,)`` int array of chosen k per env.
    """
    ks = sorted(int(k) for k in k_values)
    scores = np.asarray(scores)
    n_env = scores.shape[0]
    chosen = np.full(n_env, ks[0], dtype=np.int64)
    safe = scores <= eps  # (N, K)
    for col, k in enumerate(ks):
        chosen = np.where(safe[:, col], np.maximum(chosen, k), chosen)
    return chosen


def choose_k_greedy(
    score_fn: Callable[[int], float], k_values: list[int], eps: float
) -> tuple[int, dict, dict]:
    """Scalar (single-env / batch-shared) variant: largest safe ``k``.

    Args:
        score_fn: ``k -> float`` deviation. ``score_fn(1)`` should be 0.
        k_values: candidate rates.
        eps: threshold.

    Returns:
        ``(chosen_k, scores_by_k, safe_by_k)``.
    """
    ks = sorted(int(k) for k in k_values)
    scores = {k: float(score_fn(k)) for k in ks}
    safe = {k: (scores[k] <= eps) for k in ks}
    chosen = ks[0]
    for k in ks:
        if safe[k]:
            chosen = max(chosen, k)
    return chosen, scores, safe


# --------------------------------------------------------------------------- #
# World-model latent-deviation scoring (v0)
# --------------------------------------------------------------------------- #
def compute_latent_deviation(
    encode_fn: Callable[[Any], "np.ndarray"],
    rollout_fn: Callable[[Any, np.ndarray], "np.ndarray"],
    z0: Any,
    actions: np.ndarray,
    k_values: list[int],
    downsample_mode: str = "decimate",
) -> np.ndarray:
    """Latent-space deviation of each downsampled rollout from the k=1 rollout.

    v0 deviation (spec): roll out the original chunk ``A`` for L steps and the
    downsampled ``A^k`` for ``floor(L/k)`` steps from the same encoded start
    ``z0``; compare the A^k rollout to the A rollout at aligned indices
    ``k-1, 2k-1, ...``. ::

        deviation(k) = mean_t || z_A(t_aligned) - z_A^k(t) ||^2

    This is LATENT-SPACE deviation (not state/EEF) — acceptable for PushT v0 per
    the spec; a state-space variant can be swapped in later by replacing
    ``rollout_fn``.

    Args:
        encode_fn: unused placeholder for symmetry (z0 is already encoded);
            kept so callers can pass the encoder for future state-space scoring.
        rollout_fn: ``(z0, action_seq(T,A)) -> latent_seq(T, D)`` — predicted
            per-step latent embeddings under the given actions.
        z0: encoded current observation (opaque; passed to rollout_fn).
        actions: ``(T, A)`` original chunk for ONE env.
        k_values: candidate rates.
        downsample_mode: passed to :func:`downsample_actions`.

    Returns:
        ``(K,)`` deviation per candidate k (k=1 entry is exactly 0.0).
    """
    ks = sorted(int(k) for k in k_values)
    z_full = np.asarray(rollout_fn(z0, actions))  # (T, D)
    out = np.zeros(len(ks), dtype=np.float64)
    for i, k in enumerate(ks):
        if k == 1:
            out[i] = 0.0
            continue
        a_k = downsample_actions(actions, k, mode=downsample_mode)  # (T', A)
        z_k = np.asarray(rollout_fn(z0, a_k))  # (T', D)
        # Align: A^k step j corresponds to A step (k-1) + j*k.
        aligned = [(k - 1) + j * k for j in range(z_k.shape[0])]
        aligned = [t for t in aligned if t < z_full.shape[0]]
        m = len(aligned)
        if m == 0:
            out[i] = float("inf")  # cannot verify ⇒ unsafe
            continue
        diff = z_full[aligned] - z_k[:m]
        out[i] = float((diff ** 2).sum(axis=-1).mean())
    return out
