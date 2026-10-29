"""SpeedUpPatch-lite (SuP) policy — a world-model-gated scheduler that
accelerates a *frozen* CleanDiffuser DP by downsampling each action chunk.

Design (zero behavior change to the base DP, no edit to ``policy/dp.py``):
  ``SuPGatedDPPolicy`` subclasses ``policy.dp.DPPolicy`` and reuses its
  ``reset()`` verbatim (``super().reset()`` builds the *exact same* inner
  ``CleanDiffuserDPPolicy`` + world model). It then overrides ``get_action`` to
  drive the inner's building blocks directly — ``_update_ring`` (pixel history)
  and ``_sample_actions`` (the diffusion sampler, the ONLY RNG consumer) — while
  managing its own *per-env* action queues. The inner's own ``get_action`` /
  shared ``_buffer`` are simply bypassed, so no reblessing or ``__bases__``
  surgery is needed and the base DP class is untouched.

k=1 parity: with ``fixed_k=1`` the wrapper calls ``_update_ring`` and
``_sample_actions`` in the identical order and cadence as the base DP, consumes
RNG only through ``_sample_actions``, and decimation at k=1 is the identity — so
the executed action stream is bit-for-bit the base DP's.

Interception: when an env's queue empties, pick a rate ``k`` (fixed or WM-gated),
execute ``A^k = decimate(A, k)`` for that env, and refill its queue. Envs
schedule independently (per-env k).

Scoring (``cfg.sup.score_mode``):
  * ``"none"``           — fixed ``cfg.sup.fixed_k`` (``fixed_k=1`` = base-DP parity).
  * ``"state_adm"``      — greedy gate: largest ``k`` with ADM cost ``E(s,k) <= cfg.sup.eps``.
  * ``"state_adm_iql"``  — learned scheduler: ``argmax_k Q(s,k)``.
Both gated modes read the state-space ADM (see ``policy/sup_state_gate.py`` and
``scripts/sup/README.md``). Any gate error falls back to k=1 with a logged warning.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from policy.dp import DPPolicy
from policy.sup_utils import (
    choose_k_per_env,
    downsample_actions,
    gripper_aware_decimate,
)

_DEFAULT_SUP = {
    "enabled": True,
    "k_values": [1, 2, 3],
    "fixed_k": 2,
    "score_mode": "none",  # "none" | "state_adm" | "state_adm_iql"
    "eps": 0.05,
    "adm_path": "",   # state-space ADM ckpt (Track-A robot / Track-B full)
    "iql_path": "",   # IQL Q-net ckpt (state_adm_iql mode)
    "channel": "full",  # "robot" (faithful SuP, ADM-A) | "full" (ours, ADM-B)
    # "accumulate" = paper Eq.2 Delta (sum k deltas) — correct for PushT & cube
    # (both relative). "decimate" = paper Eq.2 Abs. "gripper_aware" = deprecated.
    "downsample_mode": "accumulate",
    "gripper_dim": -1,
    "sat_limit": 1.0,  # |merged action| > sat_limit ⇒ saturation risk (cube clips ±1)
    "log_decisions": True,
    # Added 2026-08-04. Score E(s,k) for TELEMETRY ONLY while score_mode="none"
    # keeps executing fixed_k. Needed because D' is built offline from demo
    # frames (build_iql_dataset.py replays the demo h5 every 15 frames), while
    # a rollout starts at demo frame 5 and diverges immediately — so we have
    # never observed E at the states the policy actually visits, and every
    # gated run to date logged Q-values (state_adm_iql) rather than E. This is
    # the missing measurement. It cannot change behaviour: the diffusion chunk
    # is drawn before _select_k and the ADM draws no random numbers at eval, so
    # the RNG stream is untouched and SR must reproduce exactly.
    "log_scores": False,
}


class SuPGatedDPPolicy(DPPolicy):
    """Frozen CleanDiffuser DP + SuP downsampling scheduler.

    Extra construction kwarg (recipe ``sup:`` block): ``sup`` — dict, see
    ``_DEFAULT_SUP`` for keys/defaults. All base ``DPPolicy`` kwargs still apply.
    """

    def __init__(self, *args: Any, sup: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cfg = dict(_DEFAULT_SUP)
        if sup is not None:
            user = (
                OmegaConf.to_container(sup, resolve=True)
                if isinstance(sup, DictConfig)
                else dict(sup)
            )
            cfg.update({k: v for k, v in user.items() if v is not None})
        cfg["k_values"] = sorted(int(k) for k in cfg["k_values"])
        self._sup = cfg
        # SuP runtime state (init in reset)
        self._env_queues: list[collections.deque] | None = None
        self._sup_step = 0
        self._sup_log_path: Path | None = None
        self._sup_warned_wm = False

    # ------------------------------------------------------------------ #
    def reset(self, world: Any, cfg: DictConfig | None = None, eval_world: Any | None = None) -> None:
        # Build the exact same inner CleanDiffuserDPPolicy + WM as the base DP.
        super().reset(world, cfg=cfg, eval_world=eval_world)
        self._env_queues = None
        self._sup_step = 0
        self._sup_warned_wm = False

        self._sup_log_path = None
        if cfg is not None and bool(self._sup.get("log_decisions", True)):
            try:
                repo_root = Path(__file__).resolve().parents[1]
                logs_subdir = str(cfg.output.get("logs_subdir", "logs_v3_inprocess"))
                out = repo_root / cfg.output.findings_dir / logs_subdir / cfg.output.run_name
                out.mkdir(parents=True, exist_ok=True)
                self._sup_log_path = out / "sup_decisions.jsonl"
                self._sup_log_path.write_text("")  # truncate stale log
            except Exception as e:  # noqa: BLE001
                print(f"[sup] could not open decisions log ({e}); printing instead")

        # Build the WM/state gate when gated scoring is requested — or when
        # score_mode="none" but log_scores asks for E as telemetry.
        self._gate = None
        _want_gate = (self._sup["score_mode"] in ("state_adm", "state_adm_iql")
                      or bool(self._sup.get("log_scores", False)))
        if _want_gate and cfg is not None:
            try:
                self._gate = self._build_gate(cfg)
            except Exception as e:  # noqa: BLE001
                print(f"[sup] gate build failed ({type(e).__name__}: {e}); score_mode falls back to k=1")

        print(
            f"[sup] enabled: score_mode={self._sup['score_mode']} "
            f"k_values={self._sup['k_values']} fixed_k={self._sup['fixed_k']} "
            f"eps={self._sup['eps']} downsample={self._sup['downsample_mode']}"
        )

    def _build_gate(self, cfg: DictConfig):
        """Construct the state-space ADM gate (greedy) or the IQL scheduler."""
        task_name = str(cfg.task.get("task_name", cfg.task.get("name", "")))
        mode = self._sup["score_mode"]
        if mode not in ("none", "state_adm", "state_adm_iql"):
            raise ValueError(f"unknown score_mode {mode!r} (expected none|state_adm|state_adm_iql)")
        if mode == "none" and not bool(self._sup.get("log_scores", False)):
            raise ValueError("score_mode='none' builds no gate unless log_scores is set")

        from policy.sup_state_gate import StateADMGate
        # "none"+log_scores needs deviation() only; loading the Q-net would make
        # the run depend on an iql_path that fixed-k cells legitimately leave stale.
        iql = str(self._sup["iql_path"]) if mode == "state_adm_iql" else None

        # Added 2026-08-05. Checkpoints from scripts/sup/train_iql_ak.py carry
        # arch="sa_scalar": Q(o, A^k) -> scalar, one input per candidate k, as in
        # the paper (Sec 4.2 / Alg.1 line 26). The stock gate builds one input and
        # reads a vector head, which is the wrong shape for those weights — so
        # pick the gate from the checkpoint rather than adding a config switch
        # that could disagree with the file it points at.
        if iql:
            import torch as _t
            if _t.load(iql, map_location="cpu", weights_only=False).get("arch") == "sa_scalar":
                from policy.sup_state_gate_ak import StateADMGateAK
                return StateADMGateAK(
                    task_name=task_name, adm_path=str(self._sup["adm_path"]),
                    k_values=self._sup["k_values"],
                    eps=float(self._sup["eps"]), downsample_mode=self._sup["downsample_mode"],
                    channel=str(self._sup.get("channel", "full")),
                    clip=float(self._sup.get("sat_limit", 1.0)), iql_path=iql, device="cuda",
                )
        return StateADMGate(
            task_name=task_name, adm_path=str(self._sup["adm_path"]),
            k_values=self._sup["k_values"],
            eps=float(self._sup["eps"]), downsample_mode=self._sup["downsample_mode"],
            channel=str(self._sup.get("channel", "full")),
            clip=float(self._sup.get("sat_limit", 1.0)), iql_path=iql, device="cuda",
        )

    # ------------------------------------------------------------------ #
    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        if self._inner is None:
            raise RuntimeError("SuPGatedDPPolicy.get_action called before reset(world, cfg).")
        from eval_cleandiffuser_dp import _world_pixels_to_cd_tensor

        inner = self._inner
        cur_pixels = _world_pixels_to_cd_tensor(
            info_dict,
            use_wrist=inner._dp.n_views > 1,
            image_size=inner._dp.image_size,
            device=inner._dp.device,
        )
        inner._update_ring(cur_pixels)  # inherited pixel-history update (unchanged)
        self._sup_step += 1
        n_env = int(inner._ring.shape[0])

        if self._env_queues is None:
            self._env_queues = [collections.deque() for _ in range(n_env)]

        empties = [i for i in range(n_env) if len(self._env_queues[i]) == 0]
        if empties:
            # ONE batched diffusion draw for all envs — identical RNG cadence to
            # the base DP whenever all queues empty together (uniform k, incl.
            # the k=1 parity case). Only the empty envs are refilled.
            actions = inner._sample_actions()  # (N, buffer_max, A) tensor
            actions_np = actions.detach().cpu().numpy()
            inner._n_replans += 1

            chosen_k, scores, safe = self._select_k(actions_np, info_dict, empties)
            sat = {}
            for i in empties:
                a_dec = self._downsample(actions_np[i], int(chosen_k[i]))  # (T', A)
                sat[i] = self._sat_frac(a_dec)
                for t in range(a_dec.shape[0]):
                    self._env_queues[i].append(a_dec[t])
            self._log_decision(empties, chosen_k, scores, safe, sat, actions_np.shape[1])

        return np.stack([self._env_queues[i].popleft() for i in range(n_env)], axis=0)

    @property
    def n_replans(self) -> int:
        return int(getattr(self._inner, "_n_replans", 0)) if self._inner else 0

    # ------------------------------------------------------------------ #
    def _downsample(self, actions_1env: np.ndarray, k: int) -> np.ndarray:
        mode = self._sup["downsample_mode"]
        if mode == "gripper_aware":  # deprecated (assumed absolute gripper; wrong for cube)
            return gripper_aware_decimate(actions_1env, k, gripper_dim=int(self._sup["gripper_dim"]))
        return downsample_actions(actions_1env, k, mode=mode)  # "accumulate" | "decimate"

    def _sat_frac(self, a_dec: np.ndarray) -> float:
        """Fraction of merged-action components exceeding the action limit.

        Signals saturation: an accumulated delta larger than the actuator range
        gets clipped (cube: env clips ±1) or physically unreachable (pusht: PD +
        velocity limit), capping the achievable speedup and distorting intent.
        For pusht (no hard clip, range≈±1) it is a loose proxy, not a hard bound.
        """
        lim = float(self._sup.get("sat_limit", 1.0))
        return float((np.abs(a_dec) > lim).mean())

    def _select_k(self, actions_np: np.ndarray, info_dict: dict, empties: list[int]):
        """Return (chosen_k[env], scores[env], safe[env]) for the empty envs."""
        mode = self._sup["score_mode"]
        if mode == "none":
            fk = int(self._sup["fixed_k"])
            # log_scores: record E(s,k) at the visited state without letting it
            # influence the executed k. `scores` is telemetry; `chosen_k` stays
            # fixed_k for every env.
            dev = self._state_gate_dev(actions_np, info_dict) if self._gate is not None else None
            return ({i: fk for i in empties},
                    {i: (dev[i].tolist() if dev is not None else None) for i in empties},
                    {i: None for i in empties})

        if mode == "state_adm":  # greedy state-ADM gate (SuP MPC baseline)
            dev = self._state_gate_dev(actions_np, info_dict)
            if dev is None:
                return ({i: 1 for i in empties}, {i: None for i in empties}, {i: None for i in empties})
            picks = choose_k_per_env(dev, self._sup["k_values"], float(self._sup["eps"]))
            return ({i: int(picks[i]) for i in empties}, {i: dev[i].tolist() for i in empties},
                    {i: (dev[i] <= float(self._sup["eps"])).tolist() for i in empties})

        if mode == "state_adm_iql":  # IQL scheduler: argmax_k Q(s,k)
            kv = self._sup["k_values"]
            try:
                Q = self._gate.q_values(info_dict, actions_np)  # (N, K)
            except Exception as e:  # noqa: BLE001
                self._warn_wm(f"iql q_values failed ({type(e).__name__}: {e})")
                return ({i: 1 for i in empties}, {i: None for i in empties}, {i: None for i in empties})
            picks = {i: int(kv[int(np.argmax(Q[i]))]) for i in empties}
            return (picks, {i: Q[i].tolist() for i in empties}, {i: None for i in empties})

        raise ValueError(f"unknown sup.score_mode {mode!r}")

    def _state_gate_dev(self, actions_np: np.ndarray, info_dict: dict) -> np.ndarray | None:
        if getattr(self, "_gate", None) is None:
            self._warn_wm("no state gate built"); return None
        try:
            return self._gate.deviation(info_dict, actions_np)
        except Exception as e:  # noqa: BLE001
            self._warn_wm(f"state gate deviation failed ({type(e).__name__}: {e})"); return None

    def _warn_wm(self, msg: str) -> None:
        if not self._sup_warned_wm:
            print(f"[sup] wm_gate unavailable ({msg}); falling back to k=1")
            self._sup_warned_wm = True

    def _log_decision(self, empties, chosen_k, scores, safe, sat, chunk_len) -> None:
        if not bool(self._sup.get("log_decisions", True)):
            return
        rec = {
            "env_step": int(self._sup_step),
            "n_replans": int(self.n_replans),
            "chunk_len": int(chunk_len),
            "downsample_mode": self._sup["downsample_mode"],
            "replanned_envs": [int(i) for i in empties],
            "chosen_k": {int(i): int(chosen_k[i]) for i in empties},
            "candidate_scores": {int(i): scores.get(i) for i in empties},
            "safe_mask": {int(i): safe.get(i) for i in empties},
            "sat_frac": {int(i): round(float(sat.get(i, 0.0)), 4) for i in empties},
            "buffer_len_after": {int(i): len(self._env_queues[i]) for i in empties},
        }
        line = json.dumps(rec)
        if self._sup_log_path is not None:
            with open(self._sup_log_path, "a") as f:
                f.write(line + "\n")
        else:
            print(f"[sup] {line}")
