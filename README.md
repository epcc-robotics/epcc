# Expert-Play Contouring Control: Faster-than-Demonstration Planning from Slow Expert and Fast Play

Seunghoon Cho, Wonsuhk Jung, Sundhar Vinodh Sangeetha, Shreyas Kousik

**EPCC** · [Paper](https://arxiv.org/abs/2609.22798) · [Website](https://expert-play-contouring.netlify.app/) · Checkpoints and data (coming soon)

**TL;DR.**
EPCC learns a latent world model from slow expert demonstrations together with fast play
data, generates a latent contour of the task with a diffusion model trained on the
demonstrations, and tracks that contour with model-predictive contouring control (MPCC) in the
world model's latent space. The controller can traverse the contour faster than the
demonstrations it was learned from. The code reproduces the paper's PushT, Slippery-PushT and
Cube experiments, its baselines (Diffusion Policy, SkiP, SAIL, SuP, Play+Goal), and the training
of every model involved.

## News
- 2026-10-29: code release. Checkpoints and data will follow on Hugging Face.

## Installation

Tested with Python 3.10, PyTorch 2.3.1 (CUDA 12.1) and one NVIDIA GPU.

```bash
git clone --recursive https://github.com/epcc-robotics/epcc.git epcc && cd epcc
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements-frozen.txt --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu121   # exact research environment
pip install --no-deps -e third_party/stable_worldmodel -e third_party/cleandiffuser
cp cluster.env.example cluster.env                              # then edit it (below)
```

`cluster.env` holds the site settings every launcher reads: `STABLEWM_HOME` (the data root),
`VENV`, an optional CUDA module, and Slurm settings that are only used with `--slurm`.
`third_party/` pins the exact versions of stable-worldmodel and CleanDiffuser (git submodules on
our forks) and AWE (vendored) that produced the paper's numbers; `third_party/README.md` lists every
change relative to upstream.

## Checkpoints and data

The checkpoints and data will be released on Hugging Face (coming soon). The release archive
contains every checkpoint and evaluation dataset the paper's evaluation cells use (world
models, contour generators, DP / SkiP / SAIL / SuP models, demonstration sets, latent caches)
and the precomputed observation/action scaler statistics (`stats/`). Point `STABLEWM_HOME` at it:

```bash
export STABLEWM_HOME=/path/to/epcc_data
```

The layout is listed in [docs/DATA.md](docs/DATA.md). The paper's run directories will be
released alongside (`published_runs/`). The world-model training corpora are not part of the
archive.

## Quick start

An evaluation **cell** is one Hydra config: task, policy, checkpoints and evaluation protocol.
Every number in the paper comes from a cell under `config/eval/cells/paper/`.

```bash
# two episodes of the Table I PushT EPCC cell, on the local GPU
paper/run_cell.sh paper/table1/pusht_epcc smoke task.num_eval=2

# the full cell (50 episodes), or submit it to Slurm
paper/run_cell.sh paper/table1/pusht_epcc
paper/run_cell.sh --slurm paper/table1/pusht_epcc

# score one or more runs
python paper/common/score_runs.py --out results/scores.json results/runs/pusht_epcc_n50
```

**Result contract.** A run writes `results/<group>/<run_name>/` with `results.json` (the harness's
per-episode metrics; its `success_rate` is not the paper's success definition), `rollout_traj.npz` (states, actions and success flags of every
episode), `meta.json` (the fully resolved configuration) and `control_ts_trace.npz`. The scorer
applies the paper's success definition (PushT tasks: the block stays at the goal for 50 steps;
Cube: grasped, not dropped and reached) and prints one row per run: success rate (%), median time
to completion over successful episodes (s) and throughput TP (SAIL Eq. 2, 1/s). Report these
numbers, not `results.json`. Policies sample from diffusion models without a fixed
seed, so a rerun gives numbers close to, not identical with, the published runs.

### Reproducing the paper

| Paper element | Cells | Command |
|---|---|---|
| Table I | `config/eval/cells/paper/table1/` (27) | `paper/table1/run_cells.sh [--slurm]` |
| Table II | `config/eval/cells/paper/table2/` (15) | `paper/table2/run_cells.sh [--slurm]` |
| Fig. 3a | `config/eval/cells/paper/fig3a/` (27) | `paper/fig3/run_cells.sh [--slurm]` |
| Fig. 4 | `config/eval/cells/paper/fig4/` (15) | `paper/fig4/run_cells.sh [--slurm]` |
| Appendix, data scaling | `config/eval/cells/paper/scaling/` (24) | `paper/scaling/run_cells.sh [--slurm]` |

`run_cells.sh` accepts a subset of cell names and Hydra overrides after `--`, for example
`paper/table1/run_cells.sh pusht_dp cube_dp -- task.num_eval=10`. Outputs go to
`results/<group>/<cell>/`. `paper/` also contains the scripts that turn run directories into
the paper's tables and figures (`paper/README.md`).

`python paper/common/check_cells.py` verifies that every paper cell still composes to the
configuration that was run (it detects edits to shared base configs).

### Planner hyper-parameters

The MPCC planner is configured in `config/eval/solver/` and the base cells in
`config/eval/cells/`. The main knobs, overridable per run:
`solver.inner_solver.num_samples` (CEM samples), `solver.inner_solver.n_steps` (CEM
iterations), `solver.inner_solver.topk`, `solver.inner_solver.nu_clamp` (progress-rate bounds),
`solver.reference.c` (pace of the contour), `plan_config.horizon` and
`plan_config.receding_horizon`.

## Training your own models

Training launchers are Slurm batch scripts. Submit them from the repository root with
`scripts/submit.sh`, which applies `cluster.env`.

- **World model.** `train.py` (PushT, Slippery-PushT) and `train_perview_WM_cube.py` (Cube,
  agent and wrist views) with the configs in `config/train/`. The paper's runs:
  `scripts/experiments/train/train_wm_demoonly_pusht.sbatch`,
  `scripts/experiments/train/train_perview_cube_demoonly.sbatch`, Fig. 4
  `paper/fig4/cold/train_wm_f4cold.sbatch` (+ `post_f4cold.sbatch`), data scaling
  `paper/scaling/launch_sweep.sh`. Training corpora are mixed with
  `scripts/data/build_corpus_mix.py` from the manifests in `scripts/data/config/mix/`.
- **Contour generator (DTG).** Cache demonstration latents with the trained world model
  (PushT and Slippery-PushT: `scripts/cache_latents.py --dataset pusht`, as in
  `scripts/experiments/encode_cache/encode_slippery_latents.sbatch`; Cube:
  `scripts/cache_latents_perview_aw.py`), then
  `CFG=A_mm_s1H84_lv30_plain scripts/submit.sh latent_modules/diffusion_cd/scripts/submit_train_pusht_latent_s1H84.sh`
  (Cube: `submit_train_cube_latent_s1H84.sh`, `CFG=A_mm_s1H84_lv30`). The paper selects the
  checkpoint with the lowest validation sample error after 20k steps
  (`paper/common/select_dtg_ckpt.py`).
- **Baselines.** Diffusion Policy and SkiP: `latent_modules/diffusion_cd/scripts/submit_train_dp_*.sh`.
  SAIL: labels with `scripts/sail/make_labels*.py`, then `scripts/sail/train_reached_dp*.sh` /
  `scripts/experiments/train/train_dp_reached_pusht.sbatch`. SuP: `scripts/sup/` (ADM and IQL
  gate).
- **Scaler statistics.** Cells normalize with statistics of their world model's training
  corpus. After training on new data, run `python scripts/data/cache_scaler_stats.py` (or set
  `EPCC_NO_STATS_CACHE=1` to fit them from the corpus at evaluation time).

## Data collection

The demonstration, play and rollout datasets were collected with the scripts in `scripts/data/`,
`scripts/collect_multiview_*.py` and `scripts/experiments/collect/`, as they were run for the
paper. They are provided for reference; regenerating bit-identical datasets is not guaranteed.
Use the released datasets to reproduce the paper.

## Repository layout

```
paper/                 run_cell.sh, per-element run_cells.sh, scoring and figure scripts
config/eval/cells/     evaluation cells; paper/ holds the paper's cells
config/train/          world-model training configs
scripts/baseline/      evaluation harness (eval_unified.py) and metrics
policy/                EPCC and baseline policies
latent_modules/        MPCC/MPC planners, contour generator (diffusion_cd), evaluation helpers
jepa.py, jepa_perview.py, module.py, train*.py   world models and their training
third_party/           pinned stable-worldmodel, CleanDiffuser (submodules), AWE (vendored)
docs/DATA.md           data layout under STABLEWM_HOME
```

## Citation

```bibtex
@article{cho2026epcc,
  title={Expert-Play Contouring Control: Faster-than-Demonstration Planning from Slow Expert and Fast Play},
  author={Seunghoon Cho and Wonsuhk Jung and Sundhar Vinodh Sangeetha and Shreyas Kousik},
  journal={arXiv preprint arXiv:2609.22798},
  year={2026}
}
```

## Acknowledgements and license

The code is released under the MIT License (see `LICENSE`). This code builds on [LeWorldModel](https://github.com/lucas-maes/le-wm) (MIT, see `LICENSE`),
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) (MIT),
[stable-pretraining](https://github.com/galilai-group/stable-pretraining),
[CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) (Apache-2.0) and
[AWE](https://github.com/lucys0/awe). If you use the world model, please also cite LeWorldModel:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```
