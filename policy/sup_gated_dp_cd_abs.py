"""SuP gate on a CleanDiffuser DP that emits ABSOLUTE agent waypoints.

MIRROR of ``policy/sup_gated_dp_repo.py`` — same gate, same ABS semantics, same
`waypoint_scale` conversion — with the backbone swapped from the Robomimic
DP-repo to CleanDiffuser. The original files (``sup_gated_dp_repo.py``,
``sup_gated.py``, ``sup_utils.py``, ``sup_state_gate.py``) are NOT modified, so
upstream can keep evolving them without conflicting with this.

Why a third class. There are two existing combinations and we need the missing
one:

    SuPGatedDPRepoPolicy(PushTDPRepoPolicy)   Robomimic backbone, ABS waypoints
    SuPGatedDPPolicy(DPPolicy)                CleanDiffuser backbone, delta actions
    SuPGatedDPCDAbsPolicy(DPPolicy)           CleanDiffuser backbone, ABS waypoints  <- here

Decided 2026-07-31 to move Slippery PushT off
Robomimic onto CleanDiffuser while keeping its
absolute action abstraction.

What is inherited from the CleanDiffuser side (copied from SuPGatedDPPolicy):
  - chunk acquisition: ``inner._update_ring`` + ``inner._sample_actions()``,
    which returns (N_env, T, A) — note the ENV-MAJOR layout. The DP-repo class
    works in (T, N_env, A) instead, so every index here is transposed relative
    to the file this mirrors.

What is inherited from the DP-repo side (copied from SuPGatedDPRepoPolicy):
  - ``decimate`` compression: keep every k-th absolute waypoint.
  - ``_abs_chunk_to_corpus``: the gate scores in CORPUS (delta) units, so the
    absolute waypoints are differenced and divided by ``waypoint_scale``.
    Decimating absolute waypoints is exactly accumulating the deltas between the
    kept ones, which is why the gate itself still runs in "accumulate" mode.
  - the env-space conversion at the end of ``get_action``.

🔴 TWO SCALE CONSTANTS MUST BE RE-FITTED FOR THE NEW BACKBONE
``waypoint_scale`` (px per corpus action unit) and ``env_action_scale`` (px per
env action unit) were fitted against the Robomimic DP's output convention. A
CleanDiffuser DP trained on ``pos_agent`` also emits [0,512] pixels, so they
*should* carry over unchanged — but that is an assumption until measured against
the new checkpoint. Both are config-exposed here rather than hardcoded so the
check is a config edit, not a code edit.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from policy.dp import DPPolicy
from policy.sup_gated import _DEFAULT_SUP
from policy.sup_utils import choose_k_per_env


class SuPGatedDPCDAbsPolicy(DPPolicy):
    def __init__(self, *args: Any, sup: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cfg = dict(_DEFAULT_SUP)
        cfg["downsample_mode"] = "decimate"     # ABS backbone
        cfg["waypoint_scale"] = 39.96           # px per corpus action unit
        cfg["env_action_scale"] = 100.0         # px per env action unit
        if sup is not None:
            user = (OmegaConf.to_container(sup, resolve=True)
                    if isinstance(sup, DictConfig) else dict(sup))
            cfg.update({k: v for k, v in user.items() if v is not None})
        cfg["k_values"] = sorted(int(k) for k in cfg["k_values"])
        self._sup = cfg
        self._sup_step = 0
        self._sup_log_path: Path | None = None
        self._world: Any | None = None   # set in reset(); see the note there
        self._env_queues: list[collections.deque] | None = None
        self._gate = None
        self._n_replans = 0

    def reset(self, world: Any, cfg: DictConfig | None = None, eval_world: Any | None = None) -> None:
        super().reset(world, cfg=cfg, eval_world=eval_world)
        # The class this mirrors gets `_world` from its parent
        # (pusht_dp_repo.py:157). DPPolicy does not set it, so swapping the
        # parent dropped it — and get_action needs it to read every env's live
        # agent position for the absolute -> relative conversion. Without this
        # the run dies on the first step with AttributeError (jobs 11606752-4).
        self._world = world
        self._env_queues = None
        self._sup_step = 0
        self._n_replans = 0
        self._gate = None
        # Also build for score_mode="none" when log_scores asks for E as
        # telemetry — see the note on `log_scores` in sup_gated._DEFAULT_SUP.
        if self._sup["score_mode"] != "none" or bool(self._sup.get("log_scores", False)):
            try:
                from policy.sup_state_gate import StateADMGate
                task_name = str(cfg.task.get("task_name", cfg.task.get("name", ""))) if cfg is not None else ""
                iql = str(self._sup["iql_path"]) if self._sup["score_mode"] == "state_adm_iql" else None
                # Same checkpoint-driven pick as sup_gated.py: arch="sa_scalar"
                # means Q(o, A^k) -> scalar and needs the AK gate.
                if iql:
                    import torch as _t
                    if _t.load(iql, map_location="cpu", weights_only=False).get("arch") == "sa_scalar":
                        from policy.sup_state_gate_ak import StateADMGateAK as StateADMGate
                self._gate = StateADMGate(
                    task_name=task_name, adm_path=str(self._sup["adm_path"]),
                    k_values=self._sup["k_values"], eps=float(self._sup["eps"]),
                    downsample_mode="accumulate",  # decimate-abs == accumulate-delta
                    channel=str(self._sup.get("channel", "robot")),
                    clip=float(self._sup.get("sat_limit", 1e9)), iql_path=iql, device="cuda",
                )
            except Exception as e:  # noqa: BLE001
                print(f"[sup-cd-abs] gate build failed ({type(e).__name__}: {e}); falling back to k=1")
        if cfg is not None and bool(self._sup.get("log_decisions", True)):
            try:
                repo_root = Path(__file__).resolve().parents[1]
                logs_subdir = str(cfg.output.get("logs_subdir", "logs_v3_inprocess"))
                out = repo_root / cfg.output.findings_dir / logs_subdir / cfg.output.run_name
                out.mkdir(parents=True, exist_ok=True)
                self._sup_log_path = out / "sup_decisions.jsonl"
                self._sup_log_path.write_text("")
            except Exception:  # noqa: BLE001
                self._sup_log_path = None
        print(f"[sup-cd-abs] enabled: score_mode={self._sup['score_mode']} "
              f"fixed_k={self._sup['fixed_k']} channel={self._sup.get('channel')} "
              f"waypoint_scale={self._sup['waypoint_scale']} (ABS/decimate, CleanDiffuser)")

    # ---- chunk acquisition (CleanDiffuser side) ---- #
    def _raw_chunk(self, info_dict: dict) -> np.ndarray:
        """(N_env, Ta, 2) absolute waypoints — ENV-MAJOR, unlike the DP-repo mirror."""
        from eval_cleandiffuser_dp import _world_pixels_to_cd_tensor
        inner = self._inner
        cur_pixels = _world_pixels_to_cd_tensor(
            info_dict, use_wrist=inner._dp.n_views > 1,
            image_size=inner._dp.image_size, device=inner._dp.device,
        )
        inner._update_ring(cur_pixels)
        return inner._sample_actions().detach().cpu().numpy()

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        if self._inner is None:
            raise RuntimeError("SuPGatedDPCDAbsPolicy.get_action called before reset(world, cfg).")
        raw = self._raw_chunk(info_dict)              # (N, Ta, 2) ABS px
        n_env = int(raw.shape[0])
        self._sup_step += 1

        if self._env_queues is None:
            self._env_queues = [collections.deque() for _ in range(n_env)]
        empties = [i for i in range(n_env) if not self._env_queues[i]]

        if empties:
            scores = None
            if self._sup["score_mode"] == "none":
                k = int(self._sup["fixed_k"])
                chosen_k = {i: k for i in empties}
                # log_scores: E(s,k) at the visited state, telemetry only —
                # chosen_k above is already fixed and is not revisited.
                if self._gate is not None:
                    try:
                        a_fine = self._abs_chunk_to_corpus(raw, info_dict)
                        scores = self._gate.deviation(info_dict, a_fine)   # (N,K)
                    except Exception as e:  # noqa: BLE001
                        print(f"[sup-cd-abs] log_scores deviation failed ({type(e).__name__}: {e})")
            else:
                chosen_k = self._select_k(raw, info_dict, empties)
            for i in empties:
                k = int(chosen_k[i])
                sub = raw[i, k - 1 :: k, :] if k > 1 else raw[i]   # decimate ABS waypoints
                for t in range(sub.shape[0]):
                    self._env_queues[i].append(sub[t])
            self._n_replans += 1
            if self._sup_log_path is not None:
                # `candidate_scores` mirrors the key sup_gated.py:_log_decision
                # writes, so one analysis script reads all three tasks.
                rec = {"replan": int(self._n_replans),
                       "chosen_k": {str(i): int(chosen_k[i]) for i in empties}}
                if scores is not None:
                    rec["candidate_scores"] = {str(i): scores[i].tolist() for i in empties}
                with open(self._sup_log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

        abs_action = np.stack([self._env_queues[i].popleft() for i in range(n_env)], axis=0)  # (B,2) ABS
        live_ap = np.stack([np.asarray(env.unwrapped.agent.position, dtype=np.float64)
                            for env in self._world.envs.unwrapped.envs], axis=0)
        scale = float(self._sup["env_action_scale"])
        return ((abs_action.astype(np.float64) - live_ap) / scale).astype(np.float32)

    @property
    def n_replans(self) -> int:
        return int(self._n_replans)

    # ---- gate plumbing (DP-repo side, transposed for env-major chunks) ---- #
    def _abs_chunk_to_corpus(self, raw: np.ndarray, info_dict: dict) -> np.ndarray:
        """(N,Ta,2) absolute waypoints -> (N,Ta,2) corpus agent deltas / waypoint_scale.
        a_fine[t] = (w_t - w_{t-1}) / scale, with w_{-1} = current agent position."""
        s = float(self._sup["waypoint_scale"])
        cur = np.asarray(info_dict.get("pos_agent"), np.float64)
        cur = cur[:, -1] if cur.ndim == 3 else cur                       # (N,2) latest
        prev = np.concatenate([cur[:, None, :], raw[:, :-1, :]], axis=1)  # (N,Ta,2)
        return ((raw - prev) / s).astype(np.float32)

    def _select_k(self, raw: np.ndarray, info_dict: dict, empties: list[int]) -> dict:
        """Copied verbatim from SuPGatedDPRepoPolicy._select_k — only `n_env`
        changes, because this chunk is env-major (N,Ta,A) rather than (Ta,N,A).
        Do not "simplify": the argument order is `deviation(info_dict, actions)`
        and the IQL branch uses `q_values`, not `deviation`."""
        mode = self._sup["score_mode"]
        n_env = raw.shape[0]
        if self._gate is None:
            return {i: 1 for i in range(n_env)}
        a_fine = self._abs_chunk_to_corpus(raw, info_dict)  # (N,Ta,2) corpus deltas
        try:
            if mode == "state_adm_iql":
                Q = self._gate.q_values(info_dict, a_fine)  # (N,K)
                kv = self._sup["k_values"]
                return {i: int(kv[int(np.argmax(Q[i]))]) for i in range(n_env)}
            dev = self._gate.deviation(info_dict, a_fine)   # (N,K)
            picks = choose_k_per_env(dev, self._sup["k_values"], float(self._sup["eps"]))
            return {i: int(picks[i]) for i in range(n_env)}
        except Exception as e:  # noqa: BLE001
            print(f"[sup-cd-abs] gate failed ({type(e).__name__}: {e}); k=1")
            return {i: 1 for i in range(n_env)}
