"""RolloutLogger — shared per-episode + per-step + meta logging for eval.

Plug into ANY eval script (MPC / MPCC / DP / InvDyn / SAIL / WM) by:
    1. Constructing once before the eval loop with method/task/output path.
    2. ``log_step(t, info)`` each env-step.
    3. ``log_episode_end(behavioral_results, success_flags, ...)`` after loop.
    4. ``finalize()`` to write rollout.npz + meta.yaml + keyframes.

Storage philosophy:
    - **rollout.npz**: dense per-step + per-episode arrays. Keep arrays small —
      no raw pixels, only latents (D=384) + scalars + low-dim states.
    - **meta.yaml**: human-readable provenance (method, task, ckpts, seeds,
      solver hparams). Plot script joins on (method, task) from here.
    - **keyframes/**: ``N_keyframes`` evenly-spaced PNGs per env, taken from
      the same ``video_frames`` buffer the eval loop already collected.

Latent z_t is OPTIONAL: re-encoding every executed frame is expensive, so
this is opt-in via ``attach_post_hoc_encoder(model, transform)`` + a call to
``encode_latents_posthoc(video_frames)`` after the loop.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import imageio
import numpy as np
import yaml

from latent_modules.eval.success_specs import SuccessSpec, get_success_spec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha(cwd: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _to_plain(obj: Any) -> Any:
    """Convert OmegaConf/torch/numpy types to YAML-safe primitives."""
    try:
        from omegaconf import OmegaConf, DictConfig, ListConfig
        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _evenly_spaced_indices(n: int, hi: int) -> np.ndarray:
    """Indices for ``n`` evenly-spaced keyframes in [0, hi]. Always includes
    endpoints. Returns int array of length ``n``. ``hi`` must be >= 0."""
    if hi <= 0:
        return np.zeros(n, dtype=np.int64)
    return np.linspace(0, hi, n).round().astype(np.int64)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

@dataclass
class RolloutLogger:
    """One per eval run. Holds buffers + writes rollout.npz / meta.yaml /
    keyframes/ at finalize time.

    Per-step buffers grow as ``(T_steps,)`` lists of ``(n_envs, ...)`` arrays;
    stacked to ``(n_envs, T_steps, ...)`` at finalize. Memory budget for cube
    AW (n=25, T=200, D=384, A=4, S~30): ~10 MB — fits easily.
    """

    # Required at construction
    method: str
    task: str
    out_dir: Path
    n_envs: int
    eval_budget: int

    # Per-episode anchors (filled by ``set_episodes``)
    episode_idx: np.ndarray = field(default_factory=lambda: np.empty(0))
    start_step: np.ndarray = field(default_factory=lambda: np.empty(0))
    goal_offset: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Behavioral state hooks (filled by eval script)
    state_extractor: Optional[Callable[[Any], Dict[str, Any]]] = None
    """env_unwrapped -> dict of behavioral state keys (cube_pos, effector_pos,
    gripper_opening, etc.). Called per env per step."""

    # Success specs (auto-resolved from task at __post_init__)
    success_specs: List[SuccessSpec] = field(default_factory=list)

    # Per-step running buffers
    _t: int = 0
    _actions: List[np.ndarray] = field(default_factory=list)
    _rewards: List[np.ndarray] = field(default_factory=list)
    _costs: List[np.ndarray] = field(default_factory=list)
    _theta: List[np.ndarray] = field(default_factory=list)
    _states_per_env: List[List[List[dict]]] = field(default_factory=list)
    _success_step_trace: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    _terminated_trace: List[np.ndarray] = field(default_factory=list)
    _meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if not self.success_specs:
            self.success_specs = get_success_spec(self.task)
        for spec in self.success_specs:
            self._success_step_trace[spec.name] = []
        self._states_per_env = [[] for _ in range(self.n_envs)]
        self._meta["created_at"] = datetime.now().isoformat(timespec="seconds")
        self._meta["method"] = self.method
        self._meta["task"] = self.task
        self._meta["n_envs"] = int(self.n_envs)
        self._meta["eval_budget"] = int(self.eval_budget)
        self._meta["success_criteria"] = [
            {"name": s.name, "description": s.description}
            for s in self.success_specs
        ]

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    def set_episodes(
        self,
        episode_idx: Sequence[int],
        start_step: Sequence[int],
        goal_offset: Sequence[int],
    ) -> None:
        self.episode_idx = np.asarray(list(episode_idx), dtype=np.int64)
        self.start_step = np.asarray(list(start_step), dtype=np.int64)
        self.goal_offset = np.asarray(list(goal_offset), dtype=np.int64)
        self._meta["episode_idx"] = self.episode_idx.tolist()
        self._meta["start_step"] = self.start_step.tolist()
        self._meta["goal_offset"] = self.goal_offset.tolist()

    def add_meta(self, **kwargs) -> None:
        """Drop in arbitrary YAML-safe entries. OmegaConf configs are flattened
        into plain dicts. Will overwrite existing keys."""
        for k, v in kwargs.items():
            self._meta[k] = _to_plain(v)

    def attach_demo_ttc(self, demo_ttc_path: Optional[Path]) -> None:
        """Record path to ``demo_ttc.npz`` (precomputed elsewhere). Plotter
        will load it to compute speedup_ratio per matched initial-condition."""
        if demo_ttc_path is None:
            self._meta["demo_ttc_path"] = None
            return
        p = Path(demo_ttc_path)
        self._meta["demo_ttc_path"] = str(p.resolve())
        if not p.exists():
            print(f"[RolloutLogger] WARN: demo_ttc path {p} does not exist; "
                  f"speedup_ratio will be NaN at plot time.")

    # ------------------------------------------------------------------
    # Per-step
    # ------------------------------------------------------------------

    def log_step(
        self,
        t: int,
        *,
        action: Optional[np.ndarray] = None,
        reward: Optional[np.ndarray] = None,
        cost: Optional[np.ndarray] = None,
        theta: Optional[np.ndarray] = None,
        terminated: Optional[np.ndarray] = None,
        env_states: Optional[List[dict]] = None,
    ) -> None:
        """Record one env-step worth of data.

        All arrays have leading dim = n_envs. None entries are stored as NaN
        for downstream uniformity.

        ``env_states[i]`` is the per-env state dict (already extracted) — pass
        ``cube_pos``, ``effector_pos``, ``gripper_opening``, ``goal_pos``,
        ``table_z`` for cube; pass whatever the per-task SuccessSpec expects.
        """
        assert t == self._t, f"log_step(t={t}) called out of order (expected {self._t})"
        N = self.n_envs

        def _fill(x, dim=None):
            if x is None:
                shape = (N,) if dim is None else (N, dim)
                return np.full(shape, np.nan, dtype=np.float32)
            arr = np.asarray(x)
            if arr.dtype != np.float32 and arr.dtype != np.float64:
                arr = arr.astype(np.float32)
            return arr

        self._actions.append(_fill(action) if action is None
                             else np.asarray(action, dtype=np.float32))
        self._rewards.append(_fill(reward))
        self._costs.append(_fill(cost))
        self._theta.append(_fill(theta))
        if terminated is None:
            term = np.zeros(N, dtype=bool)
        else:
            term = np.asarray(terminated, dtype=bool)
        self._terminated_trace.append(term)

        # Per-env success traces — defensive try/except per (spec, env) so a
        # single bad state dict (e.g., missing key during freeze tail) only
        # marks that one cell False instead of truncating the whole trace.
        for spec in self.success_specs:
            row = np.zeros(N, dtype=bool)
            for i in range(N):
                latest = (env_states[i] if env_states else {})
                latest = dict(latest)  # don't mutate caller's dict
                latest["t"] = t
                latest["terminated"] = bool(term[i])
                self._states_per_env[i].append(latest)
                try:
                    row[i] = bool(spec.check_step(self._states_per_env[i], latest))
                except Exception as exc:
                    # Defensive: never let a check_step raise truncate the
                    # trace. Log once per (spec, env) pair so repeated failures
                    # surface in the eval log without spamming.
                    key = (spec.name, i)
                    if not hasattr(self, "_failed_specs_seen"):
                        self._failed_specs_seen = set()
                    if key not in self._failed_specs_seen:
                        print(f"[RolloutLogger] WARN: spec={spec.name!r} env={i} "
                              f"check_step failed at t={t}: {exc!r}")
                        self._failed_specs_seen.add(key)
                    row[i] = False
                # Pop from per-env list for OTHER specs in this step — we want
                # each spec to see the same running history; share via single
                # append per step.
                if spec is not self.success_specs[-1]:
                    self._states_per_env[i].pop()
            self._success_step_trace[spec.name].append(row)

        self._t += 1

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def _compute_episode_outcomes(self) -> Dict[str, np.ndarray]:
        """For each spec: episode_success (bool) and first_success_step (int).

        first_success_step is the smallest t such that the running spec was
        True AT t. -1 if never True.
        """
        out: Dict[str, np.ndarray] = {}
        for spec in self.success_specs:
            trace = np.stack(self._success_step_trace[spec.name], axis=1)  # (N, T)
            first = np.full(self.n_envs, -1, dtype=np.int64)
            for i in range(self.n_envs):
                hits = np.flatnonzero(trace[i])
                if hits.size > 0:
                    first[i] = int(hits[0])
            out[f"success_{spec.name}"] = (first >= 0)
            out[f"first_success_step_{spec.name}"] = first
            out[f"success_t_{spec.name}"] = trace
        return out

    def _save_keyframes(
        self,
        video_frames: np.ndarray,
        first_success_step: np.ndarray,
        n_keyframes: int = 9,
    ) -> Dict[str, np.ndarray]:
        """Save ``n_keyframes`` evenly-spaced PNGs per env. Returns
        (keyframes, keyframe_steps) for inclusion in rollout.npz too.

        video_frames: (N, T, H, W, C) uint8.
        first_success_step: (N,) int — used as upper bound when >=0.
        """
        N, T = video_frames.shape[:2]
        kdir = self.out_dir / "keyframes"
        kdir.mkdir(parents=True, exist_ok=True)
        kf = np.zeros((N, n_keyframes, *video_frames.shape[2:]), dtype=np.uint8)
        kf_steps = np.zeros((N, n_keyframes), dtype=np.int64)
        for i in range(N):
            hi = int(first_success_step[i]) if first_success_step[i] >= 0 else T - 1
            hi = max(hi, 0)
            idx = _evenly_spaced_indices(n_keyframes, hi)
            kf_steps[i] = idx
            for k, t in enumerate(idx):
                kf[i, k] = video_frames[i, t]
                imageio.imwrite(kdir / f"ep_{i:03d}_{k}.png", video_frames[i, t])
        return {"keyframes": kf, "keyframe_steps": kf_steps}

    def finalize(
        self,
        *,
        video_frames: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        distances: Optional[np.ndarray] = None,
        behavioral_aggregate: Optional[Dict[str, Any]] = None,
        behavioral_per_env: Optional[List[Dict[str, Any]]] = None,
        elapsed_seconds: Optional[float] = None,
        primary_success_key: str = "naive",
        n_keyframes: int = 9,
    ) -> Dict[str, Path]:
        """Write rollout.npz + meta.yaml + keyframes. Returns paths dict.

        ``primary_success_key`` is the spec used to compute keyframes' hi
        bound and the headline TTC in meta.yaml. Plotter can re-derive any
        per-spec quantity from the npz.
        """
        N = self.n_envs
        T = self._t
        # Stack per-step buffers → (N, T, ...).
        def _stack(lst, ndim_expected: int):
            if not lst:
                return np.full((N, T) if ndim_expected == 2 else (N, T, 0),
                               np.nan, dtype=np.float32)
            return np.stack(lst, axis=1)

        actions = _stack(self._actions, ndim_expected=3)
        # Pad rewards/costs/theta to 2D
        rewards_arr = _stack(self._rewards, ndim_expected=2)
        costs_arr = _stack(self._costs, ndim_expected=2)
        theta_arr = _stack(self._theta, ndim_expected=2)
        terminated_arr = np.stack(self._terminated_trace, axis=1) if self._terminated_trace \
            else np.zeros((N, T), dtype=bool)

        # External rewards/distances override if provided (legacy compat)
        if rewards is not None and rewards.size:
            rewards_arr = np.asarray(rewards, dtype=np.float32)
        if distances is not None and distances.size:
            self._meta.setdefault("notes", []).append("distances provided externally")

        outcomes = self._compute_episode_outcomes()

        # Pick primary first-success for keyframes / TTC headline.
        primary_first = outcomes.get(
            f"first_success_step_{primary_success_key}",
            np.full(N, -1, dtype=np.int64),
        )

        kf_out: Dict[str, np.ndarray] = {}
        if video_frames is not None:
            kf_out = self._save_keyframes(
                video_frames=video_frames,
                first_success_step=primary_first,
                n_keyframes=n_keyframes,
            )

        # Headline TTC (steps) + total_steps per env.
        ttc_steps = np.where(primary_first >= 0, primary_first, self.eval_budget)
        self._meta["primary_success_key"] = primary_success_key
        self._meta["ttc_steps_mean_primary"] = float(np.mean(ttc_steps))
        self._meta["success_rate_primary"] = float(
            outcomes[f"success_{primary_success_key}"].mean() * 100
        )
        for spec in self.success_specs:
            self._meta[f"success_rate_{spec.name}"] = float(
                outcomes[f"success_{spec.name}"].mean() * 100
            )
        if elapsed_seconds is not None:
            self._meta["eval_elapsed_seconds"] = float(elapsed_seconds)
        if behavioral_aggregate is not None:
            self._meta["behavioral_aggregate"] = _to_plain(behavioral_aggregate)
        if behavioral_per_env is not None:
            self._meta["behavioral_per_env"] = _to_plain(behavioral_per_env)

        # Compose npz payload.
        payload: Dict[str, Any] = {
            "episode_idx": self.episode_idx,
            "start_step": self.start_step,
            "goal_offset": self.goal_offset,
            "actions": actions,
            "rewards": rewards_arr,
            "costs": costs_arr,
            "theta": theta_arr,
            "terminated": terminated_arr,
            "ttc_steps_primary": ttc_steps,
            "eval_budget": np.int64(self.eval_budget),
            "n_envs": np.int64(self.n_envs),
            "primary_success_key": np.array(primary_success_key),
        }
        payload.update(outcomes)
        payload.update(kf_out)

        npz_path = self.out_dir / "rollout.npz"
        np.savez_compressed(npz_path, **payload)

        meta_path = self.out_dir / "meta.yaml"
        with open(meta_path, "w") as f:
            yaml.safe_dump(_to_plain(self._meta), f, sort_keys=False, default_flow_style=False)

        print(f"[RolloutLogger] wrote {npz_path}")
        print(f"[RolloutLogger] wrote {meta_path}")
        if kf_out:
            print(f"[RolloutLogger] wrote {n_keyframes} keyframes × {N} eps "
                  f"→ {self.out_dir / 'keyframes'}")
        return {"npz": npz_path, "meta": meta_path,
                "keyframes_dir": self.out_dir / "keyframes"}
