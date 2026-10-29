"""Splice an oracle PARK tail onto pushT expert demos (Option B).

Keeps every original frame up to success UNCHANGED (stored pixels/state/action),
then for each episode reconstructs the success state in a real swm/PushT env
(_set_state + block at rest → stabilize is free, damping stops any coast), rolls
a scripted PARK policy that glides the pusher to a fixed corner for N frames
(rendering pixels live so DTG can encode them), and concatenates. Mirrors cube's
`post_success_action` (collect_multiview_diffusion_cube.py:188-217) but for pushT's
2-D relative action, without re-collecting (works for the separately collected
slippery set too — we only need the recorded success state).

Env matches the demos: PushT(relative=True, resolution=224), goal rendered.
`_set_state` sets agent pos/vel + block pos/angle but NOT block velocity (known
leak) — here that is a FEATURE: the block starts at rest at the goal.

Run via sbatch (swm env + pygame render). Args via env vars:
  SRC_H5, OUT_H5, DAMPING(0|0.35), N_PARK(default 100), CORNER(e.g. 60,60),
  SLOWDOWN(default 0.25), MAX_EP(smoke; empty=all).
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
try:
    import hdf5plugin  # noqa: F401  (must precede h5py)
except Exception:
    pass
import h5py

from stable_worldmodel.envs.pusht.env import PushT

INFO_KEYS = {"pos_agent", "vel_agent", "block_pose", "goal_pose", "goal_state",
             "goal_proprio", "coverage", "score", "n_contacts", "env_name",
             "goal", "id"}


def make_env(damping):
    env = PushT(damping=(None if damping in (0, 0.0, None) else float(damping)),
                relative=True, resolution=224)
    env.reset(seed=0)
    env.with_target = True   # ensure goal-T is rendered (matches demos)
    return env


def park_action(agent_pos, block_pos, scale, pace, retreat=150.0):
    """Retreat the pusher radially AWAY from the block AT THE DEMO PACE (no speed
    burst). Each step target = `pace` px toward a fixed retreat endpoint (150 px
    from the block on the agent's side, clamped inside walls). Moving away from
    the block guarantees increasing separation → never pushes it; capping the
    per-step move to `pace` (the episode's own median agent px/step) keeps the
    tail visually identical in speed to the slow demo. Once the endpoint is
    reached the step → 0 (agent holds)."""
    ap = np.asarray(agent_pos, np.float64); bp = np.asarray(block_pos, np.float64)
    d = ap - bp; n = np.linalg.norm(d)
    direction = d / n if n > 1e-6 else np.array([1.0, 1.0]) / np.sqrt(2.0)
    endpoint = np.clip(bp + direction * retreat, 15.0, 497.0)   # retreat target (inside walls)
    to_end = endpoint - ap; dist = np.linalg.norm(to_end)
    if dist < 1e-3:
        return np.zeros(2, np.float32)                          # reached → hold
    step = min(pace, dist)                                      # move at most `pace` px
    target = ap + (to_end / dist) * step
    a = (target - ap) / scale
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def rollout_park(env, succ_state, goal_pose, goal_state, n_park, pace):
    """Reset env to the success state and roll the demo-pace park for n_park steps."""
    env.reset(seed=0)
    # fix the goal so render_goal draws it where the demo has it
    env.goal_pose = np.asarray(goal_pose, np.float64).copy()
    env.goal_state = np.asarray(goal_state, np.float64).copy()
    env._set_state(np.asarray(succ_state, np.float64))       # agent pos/vel + block pos/angle
    env.block.velocity = (0.0, 0.0)                          # explicit: block at rest
    env.block.angular_velocity = 0.0
    recs = []
    for t in range(n_park):
        a = park_action(env.agent.position, env.block.position, env.action_scale, pace)
        obs, rew, term, trunc, info = env.step(a)
        pix = env.render()                                  # (224,224,3) uint8
        recs.append(dict(action=a, obs=obs, info=info, pixels=np.asarray(pix),
                         reward=float(rew), term=bool(term), trunc=bool(trunc)))
    return recs


def tail_col(key, recs, last_orig_val, ep_idx, last_step_idx, dtype):
    n = len(recs)
    if key == "pixels":
        return np.stack([r["pixels"] for r in recs]).astype(dtype)
    if key == "action":
        arr = np.stack([r["action"] for r in recs]).astype(dtype)
        arr[-1] = np.nan                                    # NaN final action (matches finalize convention)
        return arr
    if key == "state":
        return np.stack([np.asarray(r["obs"]["state"]) for r in recs]).astype(dtype)
    if key == "proprio":
        return np.stack([np.asarray(r["obs"]["proprio"]) for r in recs]).astype(dtype)
    if key in INFO_KEYS:
        try:
            return np.stack([np.asarray(r["info"][key]) for r in recs]).astype(dtype)
        except Exception:
            return np.repeat(np.asarray(last_orig_val)[None], n, axis=0).astype(dtype)
    if key == "terminated":
        return np.array([r["term"] for r in recs], dtype=dtype)
    if key == "truncated":
        return np.array([r["trunc"] for r in recs], dtype=dtype)
    if key == "reward":
        return np.array([r["reward"] for r in recs], dtype=dtype)
    if key == "step_idx":
        return np.arange(last_step_idx + 1, last_step_idx + 1 + n, dtype=dtype)
    if key == "ep_idx":
        return np.full(n, ep_idx, dtype=dtype)
    # default: hold the last original frame's value
    return np.repeat(np.asarray(last_orig_val)[None], n, axis=0).astype(dtype)


def main():
    src = os.environ["SRC_H5"]; out = os.environ["OUT_H5"]
    damping = float(os.environ.get("DAMPING", "0"))
    n_park = int(os.environ.get("N_PARK", "100"))
    max_ep = os.environ.get("MAX_EP", "").strip()
    max_ep = int(max_ep) if max_ep else None
    print(f"[splice] src={src}\n  out={out} damping={damping} N_PARK={n_park} "
          f"max_ep={max_ep}  (park pace = per-episode demo median agent px/step)")

    env = make_env(damping)

    with h5py.File(src, "r") as f:
        ep_len = f["ep_len"][:]; ep_off = f["ep_offset"][:]
        n_all = len(ep_len)
        n_ep = n_all if max_ep is None else min(n_all, max_ep)
        A = f["action"].shape[0]
        keys = [k for k in f.keys() if k not in ("ep_len", "ep_offset")
                and f[k].ndim >= 1 and f[k].shape[0] == A]
        print(f"[splice] {n_ep}/{n_all} eps, keys={keys}")

        new_ep_len = (ep_len[:n_ep] + n_park).astype(np.int64)
        new_ep_off = np.zeros(n_ep, np.int64); new_ep_off[1:] = np.cumsum(new_ep_len[:-1])
        total = int(new_ep_len.sum())
        print(f"[splice] total frames {A}(src) → {total}(out, +{n_park}/ep)")

        os.makedirs(os.path.dirname(out), exist_ok=True)
        with h5py.File(out, "w") as g:
            g.create_dataset("ep_len", data=new_ep_len)
            g.create_dataset("ep_offset", data=new_ep_off)
            g.attrs["spliced_park_tail"] = n_park
            g.attrs["park_mode"] = "radial_retreat_demo_pace"
            dsets = {}
            for k in keys:
                sd = f[k]
                shp = (total,) + tuple(sd.shape[1:])
                ch = (1,) + tuple(sd.shape[1:]) if sd.ndim > 1 else (min(4096, total),)
                dsets[k] = g.create_dataset(k, shape=shp, dtype=sd.dtype, chunks=ch,
                                            compression="gzip" if k == "pixels" else None)
            # re-index ep_idx cleanly if present
            for i in range(n_ep):
                o, L = int(ep_off[i]), int(ep_len[i])
                no, NL = int(new_ep_off[i]), int(new_ep_len[i])
                # 1) copy original frames verbatim
                for k in keys:
                    dsets[k][no:no + L] = f[k][o:o + L]
                # 2) roll park tail from the success (last) frame
                succ_state = f["state"][o + L - 1]
                gp = f["goal_pose"][o + L - 1]; gs = f["goal_state"][o + L - 1]
                last_step = int(f["step_idx"][o + L - 1]) if "step_idx" in keys else L - 1
                # THIS episode's demo pace (median agent px/step) → park moves at same speed
                pa_ep = f["pos_agent"][o:o + L]
                pace = (float(np.median(np.linalg.norm(np.diff(pa_ep, axis=0), axis=1)))
                        if L > 1 else 3.0)
                pace = float(np.clip(pace, 1.0, 8.0))
                recs = rollout_park(env, succ_state, gp, gs, n_park, pace)
                for k in keys:
                    last_orig = f[k][o + L - 1]
                    dsets[k][no + L:no + NL] = tail_col(
                        k, recs, last_orig, i, last_step, dsets[k].dtype)
                # clean ep_idx
                if "ep_idx" in keys:
                    dsets["ep_idx"][no:no + NL] = i
                if (i + 1) % 25 == 0:
                    print(f"[splice] ep {i+1}/{n_ep} done", flush=True)
    print(f"[splice] wrote {out}")
    # quick audit
    with h5py.File(out, "r") as g:
        el = g["ep_len"][:]
        bp = g["block_pose"]; gs = g["goal_state"]
        # last-frame block-goal dist over a few eps
        offs = np.zeros(len(el), np.int64); offs[1:] = np.cumsum(el[:-1])
        d = []
        for i in range(min(len(el), 10)):
            last = offs[i] + el[i] - 1
            d.append(float(np.linalg.norm(bp[last, :2] - gs[last, 2:4])))
        print(f"[audit] ep_len min/med/max={int(el.min())}/{int(np.median(el))}/{int(el.max())}"
              f"  last-frame block-goal dist (first 10 eps, px)={np.round(d,1).tolist()}")
    print("[done]")


if __name__ == "__main__":
    main()
