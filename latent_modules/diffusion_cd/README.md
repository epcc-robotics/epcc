# latent_diffusion_cd — CleanDiffuser-based diffusion training + eval

CleanDiffuser-based pipelines for our project:
- **`dp_cube_aw_image`** — pixel Diffusion Policy on cube AW (canonical Chi 2023 recipe).
- **`train_cube_latent_velloss`** — latent diffusion (frozen JEPA WM latent → action chunk) with 7 ablation cells (A/B/C × std/mm + canonical).

Eval scripts (also in `scripts/`) target the 3 inference pipelines: **cd_dp pixel DP**, **Diffusion-MPCC**, **Diffusion + InvDyn**, all with best-config overrides baked in.

👉 **For step-by-step paper reproduction (which ckpt → which eval → which number), see [REPRODUCE.md](REPRODUCE.md).**

Upstream CleanDiffuser is used **as a library only** (PYTHONPATH-imported, never pip-installed — see warning below). All project-specific code (dataset adapters, ε→x0 vel-loss subclass, configs, pipelines, eval planner adapter) lives in this folder + the staging root's `eval_*.py` entries.

## ⚠️ Do NOT `pip install cleandiffuser`

CleanDiffuser's `setup.py` pins `numba<0.60.0`, which transitively forces `numpy<2`. That downgrade breaks `opencv-python>=4.13` and silently corrupts torch installs that expect numpy 2.x.

Use **PYTHONPATH-based import** instead — the `scripts/setup_env.sh` script handles it:
1. fresh-clones CleanDiffuser upstream to `$CD_ROOT`
2. installs only the leaf data-deps (zarr/imagecodecs/hdf5plugin)
3. installs `numba>=0.61` (overriding the pin), preserving numpy 2.x

## Setup (one time)

```bash
# pick paths
export CD_ROOT=$HOME/CleanDiffuser                # where upstream lives
export LD_ROOT=/path/to/this/latent_diffusion_cd  # this folder
export PY=/path/to/your/venv/python               # python with torch + your repo deps

# clone + install leaf deps
bash $LD_ROOT/scripts/setup_env.sh
```

## Train

### Pixel DP (cube AW)

```bash
DATASET_PATH=/path/to/cube_multiview_diffusion_c0.3_postpad_fullscale.h5 \
PY=$PY CD_ROOT=$CD_ROOT LD_ROOT=$LD_ROOT \
sbatch $LD_ROOT/scripts/submit_train_dp_cube_aw.sh
```

Optional Hydra overrides: `gradient_steps=200000 batch_size=128 lr=2e-4`.

### Latent diffusion (cube AW)

```bash
PY=$PY CD_ROOT=$CD_ROOT LD_ROOT=$LD_ROOT \
sbatch --array=4 $LD_ROOT/scripts/submit_train_cube_latent.sh   # C_std default
```

Array index → ablation: `A_std A_mm B_std B_mm C_std C_mm B_mm_canonical`.

## Folder layout

```
latent_diffusion_cd/
├── README.md
├── REPRODUCE.md                        ← paper-number reproduction guide
├── latent_diffusion_cd/                ← importable package (`from latent_diffusion_cd.* import *`)
│   ├── dataset/
│   │   ├── cube_aw_image_dataset.py    (pixel DP adapter for OGBench cube AW)
│   │   ├── cube_latent_dataset.py      (latent input adapter)
│   │   └── normalizers.py              (StandardScaler wrapper)
│   ├── diffusion/
│   │   └── ddpm_with_vel.py            (DDPMWithVelLoss subclass — x0/eps + vel-loss head)
│   └── planner.py                      (CleanDiffuser ckpt loader + sample fn — used by eval auto-dispatch)
├── pipelines/
│   ├── utils.py                        (CleanDiffuser pipelines/utils.py + 2 patches: gym lazy import, wandb soft-fail)
│   ├── dp_cube_aw_image.py             (pixel DP train entry)
│   └── train_cube_latent_velloss.py    (latent ablation train entry)
├── configs/
│   ├── cube_latent/                    (8 yaml: base + A/B/C × std/mm + B_mm_canonical)
│   └── dp/cube_aw/chi_unet/
└── scripts/
    ├── setup_env.sh
    ├── submit_train_dp_cube_aw.sh             (train cd_dp pixel)
    ├── submit_train_cube_latent.sh            (train latent ablations — array job)
    ├── submit_eval_cd_dp_cube_aw.sh           (eval cd_dp pixel — 100% Full GTR)
    ├── submit_eval_diffusion_mpcc_cube_aw.sh  (eval Diff-MPCC — best A_mm 76%)
    └── submit_eval_diffusion_invdyn_cube_aw.sh (eval Diff+InvDyn — best C_mm 72%)
```

## What's imported from upstream CleanDiffuser (kept as library)

- `cleandiffuser.nn_diffusion.{ChiUNet1d, DiT1d, ChiTransformer}`
- `cleandiffuser.nn_condition.{MLPCondition, MultiImageObsCondition}`
- `cleandiffuser.diffusion.ddpm.DDPM` (subclassed as `DDPMWithVelLoss`)
- `cleandiffuser.dataset.dataset_utils.{MinMaxNormalizer, loop_dataloader}`
- `cleandiffuser.utils.report_parameters`
