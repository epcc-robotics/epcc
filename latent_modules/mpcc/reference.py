"""Reference trajectory in JEPA latent space with arc-length parametrization."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.interpolate as _scipy_interp
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch import nn
from torchvision.transforms import v2 as transforms


def _segment_closest(q: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor,
                     view_dims=None, view_weights=None):
    """Closest point on each segment for each query (batched).

    q:  (M, D)  query points
    p1: (K, D)  segment starts
    p2: (K, D)  segment ends

    Returns (M, K) parameter t clamped to [0,1] and squared distances.
    """
    v = p2 - p1                                                         # (K, D)
    w = q[:, None] - p1[None]                                           # (M, K, D)
    denom = (v * v).sum(-1).clamp(min=1e-8)                             # (K,)
    t = ((w * v[None]).sum(-1) / denom).clamp(0.0, 1.0)                 # (M, K)
    closest = p1[None] + t[..., None] * v[None]                         # (M, K, D)
    # t uses raw L2 geometry (closest point on segment); view weights only affect which segment wins
    if view_dims is not None and view_weights is not None:
        diff_sq = (q[:, None] - closest).pow(2)                         # (M, K, D)
        chunks = diff_sq.split(tuple(view_dims), dim=-1)                # tuple of (M, K, d_i)
        assert len(chunks) == len(view_weights), \
            f"view_dims has {len(chunks)} chunks but view_weights has {len(view_weights)} entries"
        dist_sq = sum(float(w) * c.sum(-1) for c, w in zip(chunks, view_weights))  # (M, K)
    else:
        dist_sq = (q[:, None] - closest).pow(2).sum(-1)                 # (M, K)
    return t, dist_sq


def _ensure_hwc_numpy(arr):
    """Coerce arr to numpy HWC for torchvision v2 ToImage() auto-permute.

    Identity for numpy HWC; permutes torch CHW → numpy HWC. SV legacy code
    paths that pass numpy HWC are unaffected (no shape change).
    """
    if hasattr(arr, "cpu") and hasattr(arr, "numpy"):
        arr = arr.cpu().numpy()
    if arr.ndim == 4 and arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (0, 2, 3, 1))
    return arr


class ReferenceTrajectory(nn.Module):
    """Piecewise-linear path in latent space with arc-length theta in [0, 1]."""

    def __init__(self, embeddings: torch.Tensor, theta: torch.Tensor,
                 view_dims=None, view_weights=None,
                 interpolation: str = "linear"):
        super().__init__()
        self.register_buffer("embeddings", embeddings)  # (N, D)
        self.register_buffer("theta", theta)            # (N,)
        self.N, self.D = embeddings.shape
        self.view_dims = view_dims
        self.view_weights = view_weights

        # Interpolation backend toggle. "linear" → piecewise-linear (legacy);
        # "bspline" → scipy CubicSpline (natural BC). Falls back to linear when
        # N < 4 or after deduplication leaves <4 strictly-increasing knots.
        self.interpolation = interpolation
        self._spline = None
        if interpolation == "bspline" and self.N >= 4:
            theta_np = theta.detach().cpu().numpy().astype(np.float64)
            emb_np = embeddings.detach().cpu().numpy().astype(np.float64)
            # CubicSpline requires strictly-increasing x; arc-length theta is
            # non-decreasing but may have duplicates when consecutive
            # embeddings collide. Drop duplicate-theta entries.
            keep = np.concatenate(([True], np.diff(theta_np) > 1e-8))
            theta_np = theta_np[keep]
            emb_np = emb_np[keep]
            if theta_np.size >= 4:
                self._spline = _scipy_interp.CubicSpline(
                    theta_np, emb_np, axis=0,
                    bc_type="natural", extrapolate=False,
                )
        elif interpolation not in ("linear", "bspline"):
            raise ValueError(
                f"interpolation must be 'linear' or 'bspline', got {interpolation!r}"
            )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_demonstration(
        cls,
        model: nn.Module,
        transform: Any,
        dataset_name: str,
        episode_idx: int,
        start_step: int = 0,
        end_step: int | None = None,
        chunk_size: int = 64,
        cache_dir: str | None = None,
        frameskip: int = 1,
        scalers: dict | None = None,
        view_dims: list | tuple | None = None,
        view_weights: list | tuple | None = None,
        interpolation: str = "linear",
    ) -> ReferenceTrajectory:
        """Encode an expert episode segment into a latent reference path.

        View-aware: respects ``model.enabled_views`` and ``model.proprio_keys``
        to load wrist_pixels and proprio columns alongside agent pixels.
        SV models (no ``enabled_views`` attr) default to pixels-only — same
        as legacy single-view behaviour.

        Args:
            model: encoder (JEPA / PerViewJEPA / MultiViewJEPA).
            transform: torchvision pipeline applied per pixel frame.
            dataset_name: HDF5Dataset name.
            episode_idx: episode index in the dataset.
            start_step, end_step: slice into the episode, in RAW env-step
                indices (the loader is always asked for frameskip=1).
            chunk_size: encode chunk size to bound GPU memory.
            cache_dir: HDF5Dataset cache_dir override.
            frameskip: waypoint decimation applied AFTER the raw window is
                cut. Default 1 = fine waypoints (every env-step). Set higher
                to coarsen ref granularity; the last frame is always kept.
            scalers: dict mapping proprio key → sklearn StandardScaler.
                Required for AP/AWP variants (training normalised proprio
                columns; raw values would be OOD for the ProprioEncoder).
                Ignored for SV / AW variants (no proprio).
            view_dims: list of per-view embedding dims (e.g. [192, 192] for
                agent+wrist concat). When set together with view_weights,
                `_segment_closest` re-weights per-view contributions to the
                projection distance. ``None`` → uniform (legacy behaviour).
            view_weights: list of per-view scalar weights matching
                ``view_dims`` (e.g. [1.0, 2.0] for 2× wrist weighting). Only
                used when ``view_dims`` is also set.
        """
        device = next(model.parameters()).device

        # View-aware key selection. SV (no enabled_views attr) → pixels only.
        enabled_views = tuple(getattr(model, "enabled_views", ("agent",)))
        proprio_keys = list(getattr(model, "proprio_keys", ()) or ())

        keys_to_load = ["pixels"]
        if "wrist" in enabled_views:
            keys_to_load.append("wrist_pixels")
        if "proprio" in enabled_views:
            keys_to_load.extend(proprio_keys)

        kwargs: dict[str, Any] = {
            "name": dataset_name,
            "frameskip": 1,   # raw env-steps; coarsen below (see _sel)
            "keys_to_load": keys_to_load,
        }
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        episode = swm.data.HDF5Dataset(**kwargs).load_episode(episode_idx)

        # `frameskip` coarsens the reference AFTER the raw [start, end) window
        # is cut, so start_step/end_step keep raw env-step semantics and the
        # LAST frame is always kept (theta=1 stays the goal/settle frame).
        # frameskip=1 -> _sel is None -> exact no-op vs the previous code.
        _fs = max(1, int(frameskip))
        _n = int(len(episode["pixels"][start_step:end_step]))
        _sel = None
        if _fs > 1 and _n > 0:
            _sel = list(range(0, _n, _fs))
            if _sel[-1] != _n - 1:
                _sel.append(_n - 1)
            print(f"[MPCC ref] ep={episode_idx} frameskip={_fs}: "
                  f"{_n} -> {len(_sel)} waypoints", flush=True)
        _sub = (lambda a: a) if _sel is None else (lambda a: a[_sel])

        # Pixels (always present).
        pix_raw = _ensure_hwc_numpy(_sub(episode["pixels"][start_step:end_step]))
        pix = torch.stack([transform(p) for p in pix_raw])

        # Wrist pixels (AW / AWP only).
        wrist = None
        if "wrist" in enabled_views:
            wrist_raw = _ensure_hwc_numpy(_sub(episode["wrist_pixels"][start_step:end_step]))
            wrist = torch.stack([transform(w) for w in wrist_raw])

        # Proprio (AP / AWP only). Apply scalers if provided.
        proprio_tensors: dict[str, torch.Tensor] = {}
        if "proprio" in enabled_views:
            scalers = scalers or {}
            for k in proprio_keys:
                arr = _sub(episode[k][start_step:end_step])
                if hasattr(arr, "cpu"):
                    arr = arr.cpu().numpy()
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[:, None]
                if k in scalers:
                    arr = scalers[k].transform(arr).astype(np.float32)
                proprio_tensors[k] = torch.from_numpy(arr)

        # Encode in chunks (emb_keys=[] for PreJEPA-style models with extra encoders).
        encode_kwargs: dict[str, Any] = {}
        is_prejepa = hasattr(model, "extra_encoders") and model.extra_encoders
        if is_prejepa:
            encode_kwargs["emb_keys"] = []

        embs = []
        with torch.no_grad():
            for i in range(0, len(pix), chunk_size):
                info: dict[str, Any] = {
                    "pixels": pix[i:i + chunk_size].unsqueeze(0).to(device),
                }
                if wrist is not None:
                    info["wrist_pixels"] = wrist[i:i + chunk_size].unsqueeze(0).to(device)
                for k, t in proprio_tensors.items():
                    info[k] = t[i:i + chunk_size].unsqueeze(0).to(device)
                model.encode(info, **encode_kwargs)
                emb = info["emb"].squeeze(0)  # (T, D) or (T, P, D) for PreJEPA
                embs.append(emb.cpu())
        embeddings = torch.cat(embs).to(device)

        # arc-length parametrization
        seg_lens = (embeddings[1:] - embeddings[:-1]).norm(dim=-1)
        cumul = torch.cat([seg_lens.new_zeros(1), seg_lens.cumsum(0)])
        theta = cumul / cumul[-1].clamp(min=1e-8)

        return cls(embeddings, theta, view_dims=view_dims, view_weights=view_weights,
                   interpolation=interpolation)

    @classmethod
    def from_latents(
        cls,
        embeddings: torch.Tensor,
        view_dims: list | tuple | None = None,
        view_weights: list | tuple | None = None,
        interpolation: str = "linear",
    ) -> ReferenceTrajectory:
        """Build a ReferenceTrajectory from raw latent waypoints.

        Used by diffusion-MPCC: the diffusion planner emits a sequence of
        latent waypoints directly (no dataset access). This skips the
        encoding pipeline and just computes arc-length θ ∈ [0, 1] from the
        provided embeddings.

        Args:
            embeddings: ``(N, D)`` raw latent tensor — already encoded /
                denormalized as needed. Stride/density doesn't matter; the
                arc-length parametrization absorbs it via segment lengths.

        Returns:
            ReferenceTrajectory with ``self.embeddings = embeddings`` and
            arc-length ``self.theta`` in [0, 1].
        """
        if embeddings.dim() != 2:
            raise ValueError(
                f"from_latents expects 2-D (N, D) tensor; got shape "
                f"{tuple(embeddings.shape)}"
            )
        if embeddings.shape[0] < 2:
            raise ValueError(
                f"from_latents requires N>=2 waypoints; got N={embeddings.shape[0]}"
            )
        # Same arc-length parametrization as from_demonstration.
        seg_lens = (embeddings[1:] - embeddings[:-1]).norm(dim=-1)
        cumul = torch.cat([seg_lens.new_zeros(1), seg_lens.cumsum(0)])
        theta = cumul / cumul[-1].clamp(min=1e-8)
        return cls(embeddings, theta, view_dims=view_dims, view_weights=view_weights,
                   interpolation=interpolation)

    # -- interpolation ------------------------------------------------------

    def interpolate(self, query_theta: torch.Tensor) -> torch.Tensor:
        """Interpolate latent at arc-length query.

        Backend selected by ``self.interpolation``:
          - ``"linear"`` (default): piecewise-linear via torch searchsorted.
          - ``"bspline"``: scipy CubicSpline (natural BC). Falls back to linear
            when ``self._spline is None`` (N<4 or post-dedup <4).

        query_theta: (*batch,) -> (*batch, D).
        """
        shape = query_theta.shape
        flat = query_theta.reshape(-1).clamp(0.0, 1.0)

        if self._spline is not None:
            # scipy CubicSpline path — host eval, then back to device.
            flat_np = flat.detach().cpu().numpy().astype(np.float64)
            emb_np = self._spline(flat_np)
            emb = torch.from_numpy(emb_np).to(
                device=query_theta.device, dtype=query_theta.dtype
            )
            return emb.reshape(*shape, self.D)

        idx_r = torch.searchsorted(self.theta, flat).clamp(1, self.N - 1)
        idx_l = idx_r - 1

        alpha = ((flat - self.theta[idx_l]) / (self.theta[idx_r] - self.theta[idx_l] + 1e-8))
        emb = (1 - alpha[:, None]) * self.embeddings[idx_l] + alpha[:, None] * self.embeddings[idx_r]
        return emb.reshape(*shape, self.D)

    def snap(self, query_theta: torch.Tensor) -> torch.Tensor:
        """Nearest-waypoint embedding (reachable / on-manifold target).

        Unlike ``interpolate`` (which linearly blends adjacent demo frames —
        producing off-manifold latents nothing in the data reaches), ``snap``
        returns the EXACT embedding of the nearest demonstrated waypoint. The
        target is therefore always a real demo state, reachable by an
        in-distribution action — mirroring MPC's exact-frame reference. Used by
        the ``ref_target="snap"`` cost toggle to test whether the off-manifold
        interpolated reference is what invites WM-exploitation.

        query_theta: (*batch,) -> (*batch, D).
        """
        shape = query_theta.shape
        flat = query_theta.reshape(-1).clamp(0.0, 1.0)
        idx_r = torch.searchsorted(self.theta, flat).clamp(1, self.N - 1)
        idx_l = idx_r - 1
        # pick whichever bracketing waypoint is closer in θ
        pick_r = (self.theta[idx_r] - flat) < (flat - self.theta[idx_l])
        idx = torch.where(pick_r, idx_r, idx_l)
        return self.embeddings[idx].reshape(*shape, self.D)

    # -- projection ---------------------------------------------------------

    def project(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Closest-point projection.  embeddings: (*batch, D) -> (theta, dist_sq)."""
        shape = embeddings.shape[:-1]
        flat = embeddings.reshape(-1, self.D)

        p1, p2 = self.embeddings[:-1], self.embeddings[1:]
        t, dist_sq = _segment_closest(flat, p1, p2, self.view_dims, self.view_weights)

        best_seg = dist_sq.argmin(dim=-1)
        idx = torch.arange(len(flat), device=flat.device)
        best_t = t[idx, best_seg]
        best_theta = self.theta[:-1][best_seg] + best_t * (self.theta[1:][best_seg] - self.theta[:-1][best_seg])

        return best_theta.reshape(shape), dist_sq[idx, best_seg].reshape(shape)
