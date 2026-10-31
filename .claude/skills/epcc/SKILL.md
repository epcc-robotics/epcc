---
name: epcc
description: Run, score and extend EPCC (Expert-Play Contouring Control) experiments in this repository. Use when asked to set up the environment, run an evaluation cell, reproduce a paper table or figure, score runs, tune the MPCC planner, add a new cell, or train world models, contour generators or baselines.
---

# EPCC repository guide

Read `README.md` first if it has not been read in this session. This guide adds the working rules.

## Map

| Path | Role |
|---|---|
| `config/eval/cells/` | evaluation cells (one Hydra config = task + policy + checkpoints + protocol); `paper/` holds the 108 cells behind the paper |
| `paper/run_cell.sh` | run one cell (local GPU by default, `--slurm` to submit) |
| `paper/<element>/run_cells.sh` | run all cells of a paper element (`table1`, `table2`, `fig3`, `fig4`, `scaling`) |
| `paper/common/score_runs.py` | the paper's scorer: SR, median TTC over successes, TP (SAIL Eq. 2) |
| `scripts/baseline/eval_unified.py` | evaluation harness entry point |
| `policy/`, `latent_modules/` | EPCC and baseline policies, MPCC/MPC planners, contour generator |
| `third_party/` | pinned stable-worldmodel and CleanDiffuser (submodules), AWE (vendored) |
| `cluster.env` | site settings: `STABLEWM_HOME`, `VENV`, optional `CUDA_MODULE`, Slurm settings |

## Before running anything

1. `git submodule status` shows two commits without a leading `-` (otherwise `git submodule update --init`).
2. `cluster.env` exists (copy `cluster.env.example`) and `STABLEWM_HOME` points at the `data/` folder
   of the Hugging Face release `epcc-robotics/epcc` (download command: README "Checkpoints and data");
   `$STABLEWM_HOME/stats/` holds the cached scaler statistics.
3. `python paper/common/check_cells.py` passes. It fails when a shared base config under
   `config/eval/` was edited in a way that changes a paper cell; do not "fix" it by regenerating the
   hash list unless the change is intended.

## Running cells

```bash
paper/run_cell.sh paper/table1/pusht_epcc smoke task.num_eval=2   # quick check, 2 episodes
paper/run_cell.sh paper/table1/pusht_epcc                         # 50 episodes -> results/runs/pusht_epcc_n50/
paper/run_cell.sh --slurm paper/table1/pusht_epcc                 # same, as a Slurm job
paper/table1/run_cells.sh pusht_dp cube_dp -- task.num_eval=10    # subset of an element, with overrides
```

- Arguments after the run name are Hydra overrides. A run name may not contain `/`.
- An existing run with `results.json` is never overwritten unless `OVERWRITE=1`.
- Output root: `RESULTS_DIR` (default `results/`), subdirectory `LOGS_SUBDIR` (default `runs`; the
  per-element launchers use the element name).
- `--slurm` reads the account, `GPU_PARTITION` and `GPU_GRES` from `cluster.env`; pass extra sbatch
  options with `SLURM_ARGS` (e.g. `SLURM_ARGS="--exclude=<node>"`).
- One GPU per cell. Rough wall time for 50 episodes on one RTX 6000: baselines 4-15 min, EPCC
  PushT about 30 min, Cube and Slippery-PushT about 90 min.

## Scoring and comparing

```bash
python paper/common/score_runs.py --out results/scores.json results/runs/<run> [...]
```

- Diffusion sampling is not seeded, so a rerun is close to, not identical with, the published run.
  Compare success rates with a two-sided Fisher exact test over the 50 episodes rather than by eye;
  differences of a few episodes are expected.
- Table I equal-time numbers: `paper/table1/equal_time/score_table1_eqtime.py`; Table II:
  `paper/table2/table2_final.py`. Both read published runs from `EPCC_PUBLISHED` (the
  `published_runs/` folder of the Hugging Face release) and write to `EPCC_PAPER_OUT` (default
  `results/paper`).

## Result contract

Each run directory contains `results.json` (success rate and per-episode metrics), `rollout_traj.npz`
(states, actions, success flags per episode), `meta.json` (the fully resolved configuration) and
`control_ts_trace.npz`. Read `meta.json` to learn exactly how a run was configured.

## New cells and planner tuning

- Start from the closest cell in `config/eval/cells/` (or a paper cell) and change values with
  overrides or a new yaml next to it. Do not edit base configs that paper cells inherit from.
- MPCC knobs: `solver.inner_solver.num_samples`, `solver.inner_solver.n_steps`,
  `solver.inner_solver.topk`, `solver.inner_solver.nu_clamp`, `solver.reference.c` (contour pace),
  `plan_config.horizon`, `plan_config.receding_horizon`. Change one at a time and use 50 episodes for
  any claim.

## Training

Launchers are Slurm batch scripts; submit them from the repository root with `scripts/submit.sh`.
The README section "Training your own models" lists the world-model, contour-generator and baseline
launchers. After training a world model on new data, run `python scripts/data/cache_scaler_stats.py`
(or set `EPCC_NO_STATS_CACHE=1`), otherwise cells keep using the cached statistics of the released
corpora.

## Pitfalls

- Report numbers from `paper/common/score_runs.py`, never from `success_rate` in `results.json`. The
  scorer applies the paper's success definition (PushT tasks: the block must stay at the goal for 50
  steps; Cube: grasped, not dropped and reached). The harness flag can be much higher; for example a
  Slippery-PushT SuP run reads 78% in `results.json` and 44% under the paper's definition.
- Cells normalize with the statistics of their world model's training corpus (`task.dataset_stats`),
  not of the evaluation set. A new world model needs its own statistics.
- Use `third_party/` versions only. The PyPI stable-worldmodel lacks the PushT damping, escapable
  block and exact substep rounding the paper's environments rely on, and upstream CleanDiffuser rejects
  the contour generator's horizon of 84.
