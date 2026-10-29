# SAIL — faster-than-demonstration execution

Implementation of SAIL ([arXiv:2506.11948](https://arxiv.org/abs/2506.11948))
in this repo. Measured behaviour and the gap audit against the official FastIL
repo live in `findings/260716_sail_improve/report.md`.

## What is implemented

| Paper component | Status |
|---|---|
| 4.2 controller-invariant reached-pose targets + closed-loop tracking | ✅ the load-bearing lever (k=2.5: 36% → ~92%) |
| 4.3 adaptive speed via AWE precision labels | ✅ implemented, but inert in vectorised eval (see caveats) |
| 4.1 error-adaptive guidance (EAG/CFG) | ❌ deferred |
| 4.4 action scheduling / latency | ❌ deferred |

One policy runs both evaluated arms — `policy.sail.ReachedPoseSAILPolicy`:

    use_precision=False   SAIL without the AWE scheduler (always fast)
    use_precision=True    SAIL with it (precision bit toggles slow/fast)

`policy.sail.SAILPolicy` is the naive baseline (control-rate scaling only, the
paper's DP-Fast) that those numbers are quoted against.

## Pipeline

```
  demos.h5
     |  scripts/sail/make_labels.py          reached poses + AWE/DBSCAN precision
     v
  {task}/sail/awe/*.h5                        action = [x,y,z,yaw,grip,precision]
     |  scripts/sail/train_reached_dp.sh     6-D CleanDiffuser DP, arch unchanged
     v
  {task}/sail/dp_reached/                     models/ + normalizer_stats.npz + .hydra/
     |  scripts/sail/eval_sail.sh            fair-budget k-sweep, both arms
     v
  {task}/sail/eval/                           results.json per run + sweep TSV
```

Assets follow the contract in `{beyond_demo}/{task}/sail/` — see the README
there. Scripts talk to each other only through those files; nothing imports
across stages.

```bash
# 1. labels (cube only — reached-pose extraction is task-specific)
.venv/bin/python scripts/sail/make_labels.py --workers 16

# 2. train (skips step 1 if labels already exist)
bash scripts/sail/train_reached_dp.sh
DRY_RUN=1 bash scripts/sail/train_reached_dp.sh      # plan only

# 3. evaluate both arms
bash scripts/sail/eval_sail.sh                        # without AWE scheduler
AWE=true bash scripts/sail/eval_sail.sh               # with it

# smoke
KS=2.5 REPEATS=1 N_EVAL=3 bash scripts/sail/eval_sail.sh
```

## Results (cube, n=50, fair budget)

| k | SAIL w/o AWE | SAIL w/ AWE | naive SAIL |
|---|---|---|---|
| 1.0 | — | 98%* | 92% |
| 1.5 | — | 100% | 100% |
| 2.0 | — | 92% | 64% |
| 2.5 | 90% | 88% | 36% |
| 3.0 | 92% | 92% | 16% |

\* full demo horizon (320 steps); at the shared 10 s budget it scores 84%,
where every failure is a transport truncation rather than a task failure.

The two arms are statistically indistinguishable — see caveats. pusht and
slippery_pusht have not been ported; only naive-SAIL numbers exist for them
(pusht 93.6% at k=3, slippery 78% at k=1).

## Caveats that change how you read the numbers

- **Eval is not deterministic.** DDPM sampling is not seeded per rollout; five
  identical n=50 runs at k=2.5 spanned 88-96%. Treat anything under ~4 pp as
  noise. `eval_sail.sh` runs 3 repeats by default and prints the spread.
- **The AWE scheduler is inert in vectorised eval.** 50 envs share one
  `control_timestep`, so the precision majority vote never fires and the
  observed rate stays at the fast value all sweep. Proving 4.3 needs a
  single-env lane; the w/ vs w/o AWE difference above is noise, not effect.
- **Budgets must scale with k on both knobs.** `eval_budget` *and*
  `world.max_episode_steps` — scaling one produced a spurious 0% cell before.
- **k=1 needs the demo-length horizon.** Reached-pose execution replays
  demo-speed timing, so the 10 s cross-k budget truncates it.
- **Only xyz+yaw became absolute.** The gripper channel stays commanded, same
  as FastIL.

## Layout

| file | role |
|---|---|
| `make_labels.py` | reached-pose + AWE precision label builder |
| `train_reached_dp.sh` | 6-D DP training, writes the contract layout |
| `eval_sail.sh` | fair-budget k-sweep, `AWE=true|false`, repeats + spread |

Runtime code is `policy/sail.py`; wiring is
`config/eval/baseline_corl_v2/{cube_sail_reached.yaml, policy/sail_reached.yaml}`.
