"""State-space ADM gate + IQL scheduler for SuP (online), with robot|full channel.

Reads low-dim env state from info_dict and uses a trained state-space ADM
($STABLEWM_HOME/sup/adm_*.pt) to score each candidate downsample rate k.

  channel="robot"  → ADM Track-A (robot-only): E = max agent/ee deviation   (FAITHFUL SuP)
  channel="full"   → ADM Track-B (robot+object): E = max(ee, object) deviation (OURS)

  score_mode="state_adm"      → greedy: largest k with E<=eps
  score_mode="state_adm_iql"  → IQL: argmax_k Q(s,k),  s=(env_state, chunk)

Env state (ADM's own `cols`, corpus order):
  cube            : info['observation'][:, cols]           (28-d MuJoCo obs)
  pusht/slippery  : [ax,ay,bx,by,ba,vx,vy][:, cols]        (assembled from info fields)

Errors are raised to the caller (sup_gated catches → safe k=1 fallback).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_SUP = Path(__file__).resolve().parents[1] / "scripts/sup"
if str(_SUP) not in sys.path:
    sys.path.insert(0, str(_SUP))
from adm_model import ADM  # noqa: E402


def _mlp(inp, out, h=256):
    return nn.Sequential(nn.Linear(inp, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, out))


class _QNet(nn.Module):
    def __init__(self, sd, nk):
        super().__init__(); self.q1 = _mlp(sd, nk); self.q2 = _mlp(sd, nk)
    def forward(self, s): return self.q1(s), self.q2(s)


class StateADMGate:
    def __init__(self, task_name: str, adm_path: str, k_values: list[int], eps: float,
                 downsample_mode: str, channel: str = "full", clip: float = 1.0,
                 iql_path: str | None = None, device: str = "cuda") -> None:
        self.task = task_name
        self.is_cube = task_name.endswith("cube")
        self.channel = channel
        self.k_values = sorted(int(k) for k in k_values)
        self.eps = float(eps); self.mode = downsample_mode; self.clip = float(clip); self.device = device
        self._dumped = False

        d = torch.load(adm_path, map_location=device, weights_only=False)
        self.adm = ADM(d["sd"], d.get("adim", 2), hidden=d["hidden"]).to(device)
        self.adm.load_state_dict(d["state_dict"]); self.adm.eval()
        self.sm = np.asarray(d["s_mean"], np.float32); self.ss = np.asarray(d["s_std"], np.float32)
        self.am = np.asarray(d["a_mean"], np.float32); self.asd = np.asarray(d["a_std"], np.float32)
        self.cols = list(d["cols"]); self.sd = d["sd"]; self.adim = d.get("adim", 2)
        # agent(ee) dims are the LEADING dims of every track's state vector.
        n_ag = 3 if self.is_cube else 2
        self.agent_sl = slice(0, n_ag)
        # object dims present only in Track-B (full): cube block xyz at 5:8, pt block xy at 4:6.
        if channel == "full":
            self.obj_sl = slice(5, 8) if self.is_cube else slice(4, 6)
        else:
            self.obj_sl = None

        self.iql = None
        if iql_path:
            ck = torch.load(iql_path, map_location=device, weights_only=False)
            self.q = _QNet(ck["sd"], len(ck["KS"])).to(device); self.q.load_state_dict(ck["q"]); self.q.eval()
            self.q_sm = np.asarray(ck["s_mean"], np.float32); self.q_ss = np.asarray(ck["s_std"], np.float32)
            self.iql_ks = list(ck["KS"])
            self.iql = self.q          # marks "IQL loaded" for the banner below
        print(f"[state-gate] task={task_name} channel={channel} sd={self.sd} adim={self.adim} "
              f"eps={eps} mode={downsample_mode} iql={'yes' if self.iql else 'no'}")

    def _state(self, info: dict) -> np.ndarray:
        def latest(x):
            x = np.asarray(x, np.float32); return x[:, -1] if x.ndim == 3 else x
        if self.is_cube:
            obs = info.get("observation", None)
            if obs is None:
                raise KeyError("cube state-gate needs info['observation']")
            return latest(obs)[:, self.cols].astype(np.float32)
        ag = latest(info["pos_agent"]); vel = latest(info["vel_agent"]); bp = latest(info["block_pose"])
        full = np.concatenate([ag[:, :2], bp[:, :3], vel[:, :2]], axis=1)  # corpus order [ax,ay,bx,by,ba,vx,vy]
        return full[:, self.cols].astype(np.float32)

    def _merge(self, A, k):
        if k == 1:
            return A
        l = A.shape[0] // k
        if l < 1:
            return None
        if self.mode == "decimate":
            return A[k - 1::k][:l]
        return np.clip(A[:l * k].reshape(l, k, A.shape[1]).sum(1), -self.clip, self.clip)

    @torch.no_grad()
    def _rollout(self, s0, acts):
        H = acts.shape[1]
        s0t = torch.as_tensor((s0 - self.sm) / self.ss, dtype=torch.float32, device=self.device)
        awt = torch.as_tensor((acts - self.am) / self.asd, dtype=torch.float32, device=self.device)
        out = self.adm.rollout_multi(s0t, awt, list(range(1, H + 1)))
        ss_t = torch.as_tensor(self.ss, device=self.device)
        return torch.stack([torch.as_tensor(s0, device=self.device) + out[h] * ss_t for h in range(1, H + 1)], dim=1).cpu().numpy()

    @torch.no_grad()
    def deviation(self, info_dict: dict, actions_np: np.ndarray) -> np.ndarray:
        s0 = self._state(info_dict)
        N = actions_np.shape[0]; A = actions_np[:, :, :self.adim].astype(np.float32); T = A.shape[1]
        if not self._dumped:
            self._dumped = True
            print(f"[state-gate] first call: channel={self.channel} N={N} T={T} s0={s0.shape}")
        fine = self._rollout(s0, A)
        dev = np.zeros((N, len(self.k_values)), np.float64)
        for ci, k in enumerate(self.k_values):
            if k == 1:
                continue
            l = T // k
            if l < 1:
                dev[:, ci] = 1e9; continue
            merged = np.stack([self._merge(A[e], k) for e in range(N)], axis=0)
            pm = self._rollout(s0, merged)
            idx = [min((j + 1) * k - 1, T - 1) for j in range(l)]
            ee = np.linalg.norm(fine[:, idx, self.agent_sl] - pm[:, :, self.agent_sl], axis=2)
            if self.obj_sl is not None:
                ob = np.linalg.norm(fine[:, idx, self.obj_sl] - pm[:, :, self.obj_sl], axis=2)
                dev[:, ci] = np.maximum(ee, ob).max(axis=1)
            else:
                dev[:, ci] = ee.max(axis=1)
        return dev

    @torch.no_grad()
    def q_values(self, info_dict: dict, actions_np: np.ndarray) -> np.ndarray:
        s0 = self._state(info_dict)
        N, T = actions_np.shape[0], actions_np.shape[1]
        chunk = actions_np[:, :, :self.adim].reshape(N, -1).astype(np.float32)
        s = np.concatenate([s0, chunk], axis=1); exp = self.q_sm.shape[0]
        if s.shape[1] != exp:
            # eval chunk length (T*adim) can differ from the IQL's train-time C*adim
            # (e.g. DP-repo Ta=16 vs C=15). We pad/truncate the chunk tail to match, but
            # WARN once — a large mismatch means the IQL was trained at a different chunk
            # length and its Q is being read on a re-shaped state (see KNOWN_ISSUES.md).
            if not getattr(self, "_warned_qdim", False):
                self._warned_qdim = True
                print(f"[state-gate] WARN q_values state dim {s.shape[1]} != IQL train dim {exp} "
                      f"(chunk T*adim={chunk.shape[1]}); pad/truncate tail. Re-train IQL at eval chunk len to fix.")
            s = np.concatenate([s, np.zeros((N, exp - s.shape[1]), np.float32)], axis=1) if s.shape[1] < exp else s[:, :exp]
        st = torch.as_tensor((s - self.q_sm) / self.q_ss, dtype=torch.float32, device=self.device)
        q1, q2 = self.q(st)
        return torch.minimum(q1, q2).cpu().numpy()
