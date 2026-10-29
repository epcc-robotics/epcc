"""Per-episode demo TTC (time-to-completion) — two entry points.

The plotting script needs ``TTC_demo`` for each matched eval episode to build
the speedup ratio (TTC_demo / TTC_eval) in Sec 7.4 and Ablation B's α_i in
Sec 7.7.

The **default** entry point is ``compute_demo_ttc_for_run(run_dir, criterion)``
called from ``plot_main.py``. It reads the run's ``meta.yaml``, opens the
dataset, and scans ONLY the matched ``episode_idx`` windows (typically
N_eval=10-50 episodes instead of the full 800+ in the dataset). The result is
cached as ``<run_dir>/demo_ttc__<criterion>.npz`` so re-plotting is instant.

The **CLI** entry point (``python -m latent_modules.eval.demo_ttc``) scans an entire
dataset — kept as an escape hatch for when you want to re-score every demo
under a different criterion (e.g., sweeping GTR thresholds).

Usage (CLI):
    python -m latent_modules.eval.demo_ttc \\
        --dataset "Multiview WM/cube_multiview_diffusion" \\
        --task cube-mv --criterion gtr
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

import stable_worldmodel as swm

from latent_modules.eval.success_specs import get_success_spec


def _episode_state_seq(
    episode: dict, task: str
) -> List[dict]:
    """Extract per-step behavioral state dicts for one episode.

    For cube AW: cube_pos comes from the qpos/object slice if present, or
    from privileged columns. Goal pos + table z are constants per episode.
    """
    n = len(next(iter(episode.values())))
    states: List[dict] = []

    # Cube-specific extraction.
    cube_pos_arr = None
    eff_pos_arr = None
    grip_arr = None
    goal_pos = None
    table_z = None

    # Try privileged columns first.
    for k in ("privileged_block_0_pos", "object_pos", "cube_pos"):
        if k in episode:
            cube_pos_arr = np.asarray(episode[k])
            break
    for k in ("proprio/effector_pos", "proprio_effector_pos", "effector_pos"):
        if k in episode:
            eff_pos_arr = np.asarray(episode[k])
            break
    for k in ("proprio/gripper_opening", "proprio_gripper_opening", "gripper_opening"):
        if k in episode:
            grip_arr = np.asarray(episode[k]).reshape(-1)
            break
    # Goal: try canonical fork-side names first, then the cube AW dataset's
    # ``privileged_target_block_pos`` (constant per episode, but loaded as a
    # (T, 3) array so we slice frame 0).
    for k in ("goal_privileged_block_0_pos", "goal_block_0_pos", "target_pos",
              "privileged_target_block_pos"):
        if k in episode:
            v = np.asarray(episode[k])
            goal_pos = v[0] if v.ndim > 1 else v
            break
    # Table z heuristic: cube_z at step 0 (cube starts on table).
    if cube_pos_arr is not None:
        table_z = float(cube_pos_arr[0, 2])

    # Demo-side ``terminated`` (when the column exists): used directly by the
    # ``naive`` success spec for tasks like PushT where the env's IoU/state
    # threshold fires in the recorded demo data.
    terminated_arr = None
    for k in ("terminated", "done", "ep_physical_success"):
        if k in episode:
            v = np.asarray(episode[k])
            if v.shape[0] == n and v.dtype == bool:
                terminated_arr = v
                break

    for t in range(n):
        s = {"t": t}
        if cube_pos_arr is not None:
            s["cube_pos"] = cube_pos_arr[t]
        if eff_pos_arr is not None:
            s["effector_pos"] = eff_pos_arr[t]
        if grip_arr is not None:
            s["gripper_opening"] = float(grip_arr[t])
        if goal_pos is not None:
            s["goal_pos"] = goal_pos
        if table_z is not None:
            s["table_z"] = table_z
        s["terminated"] = bool(terminated_arr[t]) if terminated_arr is not None else False
        states.append(s)
    return states


def _scan_episode_for_ttc(states: list, spec) -> int:
    """Run ``spec.check_step`` incrementally over an ordered list of state
    dicts and return the smallest t where the criterion first holds, else -1.

    The list is consumed AS-IF the agent were executing it: at each t we
    append ``states[t]`` to the running history and call ``check_step``."""
    running: list = []
    for t, latest in enumerate(states):
        running.append(latest)
        try:
            if spec.check_step(running, latest):
                return int(t)
        except Exception:
            continue
    return -1


def compute_demo_ttc(
    dataset_name: str,
    task: str,
    criterion: str,
    cache_dir: Path | None = None,
    episode_filter: list | None = None,
) -> dict:
    """Scan demo episodes; for each, find first step where ``criterion`` holds.

    ``episode_filter`` is a list of episode indices to score (subset mode).
    When None, scans every episode in the dataset (full mode — CLI default).

    Returns dict with: episode_idx (E,), ttc_steps (E,), ep_len (E,),
    criterion, task, dataset. ttc_steps = -1 if criterion never held.
    """
    specs = get_success_spec(task)
    spec_by_name = {s.name: s for s in specs}
    if criterion not in spec_by_name:
        raise KeyError(f"Task {task!r} has no spec {criterion!r}. "
                       f"Available: {list(spec_by_name)}")
    spec = spec_by_name[criterion]

    cache_dir = Path(cache_dir) if cache_dir else Path(swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name, keys_to_cache=[], cache_dir=cache_dir,
    )
    cols = dataset.column_names

    ep_col = "episode_idx" if "episode_idx" in cols else "ep_idx"
    full_ep_indices = np.unique(dataset.get_col_data(ep_col)).astype(np.int64)

    if episode_filter is not None:
        wanted = np.asarray(episode_filter, dtype=np.int64)
        ep_indices = np.array(
            [e for e in wanted if e in set(full_ep_indices.tolist())],
            dtype=np.int64,
        )
        missing = [int(e) for e in wanted if e not in set(full_ep_indices.tolist())]
        if missing:
            print(f"[demo_ttc] WARN: requested episodes not in dataset: {missing}")
    else:
        ep_indices = full_ep_indices
    print(f"[demo_ttc] dataset={dataset_name}  task={task}  criterion={criterion}  "
          f"n_episodes={len(ep_indices)}"
          f"{'  (filtered)' if episode_filter is not None else ''}")

    ttc_steps = np.full(len(ep_indices), -1, dtype=np.int64)
    ep_lens = np.zeros(len(ep_indices), dtype=np.int64)

    for i, ep_idx in enumerate(ep_indices):
        episode = dataset.load_episode(int(ep_idx))
        states = _episode_state_seq(episode, task)
        ep_lens[i] = len(states)
        ttc_steps[i] = _scan_episode_for_ttc(states, spec)
        if i < 5 or (episode_filter is None and i % 50 == 0):
            print(f"  ep {ep_idx}: len={ep_lens[i]:4d}, ttc={ttc_steps[i]}")

    return {
        "episode_idx": ep_indices,
        "ttc_steps": ttc_steps,
        "ep_len": ep_lens,
        "criterion": np.array(criterion),
        "task": np.array(task),
        "dataset": np.array(dataset_name),
    }


def compute_demo_ttc_for_run(
    run_dir: Path,
    criterion: str,
    force: bool = False,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Compute demo TTC for the episodes a SPECIFIC eval run touched.

    Reads ``run_dir/meta.yaml`` to get the dataset name, task, and matched
    ``episode_idx`` + ``start_step``. Scans only those demos. Caches the
    result to ``run_dir/demo_ttc__<criterion>.npz`` so subsequent re-plots
    are instant.

    Anchoring: the demo TTC is measured **from the same start_step the eval
    used** — both numerator (TTC_demo) and denominator (TTC_eval) share the
    same anchor, so the ratio is offset-invariant (fair across methods that
    start at different absolute frames).

    Returns:
        np.ndarray of shape (n_envs,) int64; -1 where the criterion was not
        satisfied within the demo window.
    """
    import yaml as _yaml
    run_dir = Path(run_dir)
    cache_path = run_dir / f"demo_ttc__{criterion}.npz"
    if cache_path.exists() and not force:
        cached = dict(np.load(cache_path, allow_pickle=True))
        return cached["ttc_steps_per_env"]

    meta_path = run_dir / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing — was rollout written?")
    with open(meta_path) as f:
        meta = _yaml.safe_load(f) or {}

    task = str(meta.get("task", "unknown"))
    dataset_name = meta.get("eval_dataset")
    if dataset_name is None:
        raise KeyError("meta.yaml lacks eval_dataset")
    episode_idx = np.asarray(meta.get("episode_idx", []), dtype=np.int64)
    start_step = np.asarray(meta.get("start_step", []), dtype=np.int64)
    if episode_idx.size == 0:
        raise ValueError("meta.yaml has empty episode_idx")

    # Resolution order for cache_dir:
    #   1. caller-supplied (e.g., from plot CLI flag)
    #   2. meta.yaml cache_dir (recorded by eval at write time — works from
    #      any host with the same NFS layout)
    #   3. swm default (~/.stable_worldmodel or $STABLEWM_HOME)
    if cache_dir is None and meta.get("cache_dir"):
        cache_dir = Path(meta["cache_dir"])

    # Score each requested episode independently — we may have duplicates
    # if eval re-used an episode across envs, so we can't dedupe.
    specs = get_success_spec(task)
    spec_by_name = {s.name: s for s in specs}
    if criterion not in spec_by_name:
        raise KeyError(f"Task {task!r} has no spec {criterion!r}. "
                       f"Available: {list(spec_by_name)}")
    spec = spec_by_name[criterion]

    cache_dir = Path(cache_dir) if cache_dir else Path(swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name, keys_to_cache=[], cache_dir=cache_dir,
    )

    # Cache full-episode state seqs keyed by ep_idx so duplicate episodes
    # don't reload from disk.
    seq_cache: dict[int, list] = {}
    ttc_per_env = np.full(len(episode_idx), -1, dtype=np.int64)
    for i, (ep_idx, s0) in enumerate(zip(episode_idx, start_step)):
        ep_key = int(ep_idx)
        if ep_key not in seq_cache:
            episode = dataset.load_episode(ep_key)
            seq_cache[ep_key] = _episode_state_seq(episode, task)
        seq = seq_cache[ep_key]
        # Anchor TTC at the eval's start frame: trim the demo to [s0:].
        trimmed = seq[int(s0):]
        ttc_per_env[i] = _scan_episode_for_ttc(trimmed, spec)
        if i < 3:
            print(f"  [demo_ttc] env {i}: ep={ep_key} start={int(s0)} "
                  f"len={len(trimmed)} ttc={ttc_per_env[i]}")

    out = {
        "ttc_steps_per_env": ttc_per_env,
        "episode_idx": episode_idx,
        "start_step": start_step,
        "criterion": np.array(criterion),
        "task": np.array(task),
        "dataset": np.array(dataset_name),
    }
    np.savez(cache_path, **out)
    print(f"[demo_ttc] wrote {cache_path}  "
          f"({(ttc_per_env >= 0).sum()}/{len(ttc_per_env)} eps scored)")
    return ttc_per_env


def _default_out_path(cache_dir: Path, dataset_name: str, criterion: str) -> Path:
    safe = dataset_name.replace("/", "__")
    return cache_dir / f"demo_ttc__{safe}__{criterion}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="dataset name passed to swm.data.HDF5Dataset")
    ap.add_argument("--task", required=True,
                    help="task key for success_specs lookup (e.g., cube-mv)")
    ap.add_argument("--criterion", default="gtr",
                    help="success spec name (default: gtr)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default=None, help="output npz path (default under cache dir)")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(swm.data.utils.get_cache_dir())
    out = compute_demo_ttc(args.dataset, args.task, args.criterion, cache_dir=cache_dir)

    out_path = Path(args.out) if args.out else _default_out_path(cache_dir, args.dataset, args.criterion)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"[demo_ttc] wrote {out_path}")
    print(f"  n_eps                 = {len(out['episode_idx'])}")
    valid = out["ttc_steps"] >= 0
    print(f"  ttc valid              = {valid.sum()}/{len(valid)}")
    if valid.any():
        print(f"  ttc steps mean/median = {out['ttc_steps'][valid].mean():.1f} / "
              f"{np.median(out['ttc_steps'][valid]):.1f}")
    print(f"  ep_len  mean/median   = {out['ep_len'].mean():.1f} / "
          f"{np.median(out['ep_len']):.1f}")


if __name__ == "__main__":
    main()
