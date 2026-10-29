# Reproduction guide — CleanDiffuser experiments (cube AW, c=0.3)

End-to-end recipe for the 5 best ckpts in the paper. Each `sbatch` command has the best hyperparameters baked in as defaults — you only need to point it at the right ckpt path.

## TL;DR — what gets reproduced

All eval numbers are on the **matched protocol**: 15 env-steps per replan (MPCC: `stride=5 × rh=3`; cd_dp: `stride=1 × rh=15`).

| Setting | Train job | Eval | Best SR (Full GTR, n=50) |
|---|---|---|---|
| **cd_dp pixel DP** (action diffusion) | `submit_train_dp_cube_aw.sh` | `submit_eval_cd_dp_cube_aw.sh` | **100%** ⭐⭐ |
| **A_mm** ChiUNet + x0 + MinMax (latent) | `submit_train_cube_latent.sh --array=1` | `submit_eval_diffusion_mpcc_cube_aw.sh` | **76%** ⭐ |
| **C_mm** DiT + x0 + MinMax (latent) | `submit_train_cube_latent.sh --array=5` | `submit_eval_diffusion_invdyn_cube_aw.sh` | **72%** ⭐ |
| **C_std** DiT + x0 + Standard (latent) | `submit_train_cube_latent.sh --array=4` | both mpcc/invdyn | 66 / 70% |
| **A_std** ChiUNet + x0 + Standard (latent) | `submit_train_cube_latent.sh --array=0` | both | 62 / 46% |

`B_*` (ε-prediction) variants are **broken** for latent diffusion — skip indices 2, 3, 6.

## 0. One-time setup

```bash
# 1. Clone this repo (latent_diffusion_cd lives at its root).
# 2. Set env paths:
export STAGING_ROOT=/path/to/repo        # this clone's root
export LD_ROOT=$STAGING_ROOT/latent_diffusion_cd
export CD_ROOT=$HOME/CleanDiffuser                              # pristine upstream
export PY=/path/to/your/venv/python                             # python with torch + your le-wm/swm env
export STABLEWM_HOME=/path/to/stable-wm                         # cache_dir for WM ckpts + H5

# 3. Clone upstream CleanDiffuser + install leaf deps (NOT pip install cleandiffuser).
bash $LD_ROOT/scripts/setup_env.sh

# 4. Provide a trained WM checkpoint (PerView JEPA, cube AW) — for example:
#    POLICY=pv_aw_cube_scratch_wm_training/lewm_pv_aw_cube_scratch_epoch_11
#    The eval scripts resolve this relative to $STABLEWM_HOME.
```

## 1. Training (5 ckpts)

### 1.1 cd_dp pixel DP (action diffusion)

```bash
DATASET_PATH=$STABLEWM_HOME/Multiview\ WM/cube_multiview_diffusion_c0.3_postpad_fullscale.h5 \
sbatch $LD_ROOT/scripts/submit_train_dp_cube_aw.sh
```

→ `<work_dir>/dp_cube_aw_image/cube_aw_chi_unet_ddpm_step50_h16_<TS>/42/models/model_90000.pt`

### 1.2 Latent ablations (4 ckpts in one array submit)

Array index → ablation: `0=A_std  1=A_mm  4=C_std  5=C_mm` (skip 2/3/6 = ε-pred B variants, broken).

```bash
# All four at once (needs 4 GPUs):
sbatch --array=0,1,4,5%4 $LD_ROOT/scripts/submit_train_cube_latent.sh

# Or wave-split (StandardScaler vs MinMax):
sbatch --array=0,4%2 $LD_ROOT/scripts/submit_train_cube_latent.sh   # Standard
sbatch --array=1,5%2 $LD_ROOT/scripts/submit_train_cube_latent.sh   # MinMax
```

Output paths per ablation (under `<work_dir>/cube_latent_velloss/`):
- `A_std_chiunet_x0_standard_<TS>/42/models/model_20000.pt`  *(best step from val)*
- `A_mm_chiunet_x0_minmax_<TS>/42/models/model_30000.pt`
- `C_std_dit_x0_standard_<TS>/42/models/model_40000.pt`
- `C_mm_dit_x0_minmax_<TS>/42/models/model_50000.pt`

Pick `model_<best_step>.pt` per the val_sample_l2 table:

| ablation | best step | val_sample_l2 |
|---|---|---|
| A_std | 20000 | 0.294 |
| A_mm  | 30000 | 0.023 |
| C_std | 40000 | 0.618 |
| C_mm  | 50000 | 0.074 |
| cd_dp | 90000 | 0.026 |

## 2. Eval (5 jobs)

All eval scripts:
- auto-dispatch between **legacy** (`VelLossDiffusionSDE` ckpt) and **CleanDiffuser** (sibling `meta.json`) ckpt formats — no flag needed
- have **best overrides baked in** as defaults (n_eval=50, rh=3, etc.) — pass `DIFF_CKPT=` and submit
- live in `$LD_ROOT/scripts/`

### 2.1 cd_dp pixel DP → 100% (Full GTR, n=50, matched rh=15)

```bash
DIFF_CKPT=<dp_cube_aw_image>/<exp>/42/models/model_90000.pt \
POLICY=$POLICY \
sbatch $LD_ROOT/scripts/submit_eval_cd_dp_cube_aw.sh
```

Default `PLAN_RH=15` ensures 15 env-steps per replan, matching MPCC's `stride=5 × rh=3`. (cd_dp also hits 100% at rh=3, but that's 5× more replans than MPCC — the matched protocol is the fair comparison.)

**Action normalizer**: `dp_image.py` dumps `normalizer_stats.npz` alongside each ckpt at train start, and eval loads it from there. `TRAIN_H5` is no longer required — set it only as a fallback for legacy ckpts that don't have the sibling stats file.

### 2.2 Diffusion-MPCC (4 latent ckpts)

```bash
# A_mm → 76% ⭐
DIFF_CKPT=<cube_latent_velloss>/A_mm_chiunet_x0_minmax_<TS>/42/models/model_30000.pt \
POLICY=$POLICY \
sbatch $LD_ROOT/scripts/submit_eval_diffusion_mpcc_cube_aw.sh

# Same command, swap DIFF_CKPT for: A_std step20k (62%), C_std step40k (66%), C_mm step50k (52%).
```

### 2.3 Diffusion + InvDyn (4 latent ckpts)

```bash
# C_mm → 72% ⭐
DIFF_CKPT=<cube_latent_velloss>/C_mm_dit_x0_minmax_<TS>/42/models/model_50000.pt \
INVDYN_CKPT=<your-trained-invdyn>/best.pt \
POLICY=$POLICY \
sbatch $LD_ROOT/scripts/submit_eval_diffusion_invdyn_cube_aw.sh

# Same command, swap DIFF_CKPT for: A_std step20k (46%), A_mm step30k (64%), C_std step40k (70%).
```

## 3. Known gotchas + fixes

1. **`B_*` (ε-prediction) variants broken** on latent diffusion (~0% SR). Don't run array indices 2, 3, 6.
2. **MinMax (`*_mm`) ckpts may be missing `normalizer_stats.npz`** if the train pipeline patch wasn't applied. Fix already in `pipelines/train_cube_latent_velloss.py` (single-line guard removal). Verify `normalizer_stats.npz` exists in each `42/` dir after training.
3. **cd_dp eval requires `solver.inner_solver.physical_dim=5`** — baked into `submit_eval_cd_dp_cube_aw.sh` as default. Don't override unless the dataset's action dim changes.
4. **Diff+InvDyn `plan_config.receding_horizon=3`** — baked in. Default `rh=5` gives a 16 pp gap on the matched protocol.
5. **Replan-rate matching across eval routes**: MPCC and Diff+InvDyn use `stride=5, rh=3` → 15 env-steps per replan. cd_dp uses `stride=1, action_block=1, rh=15` for the same 15-step cadence. Don't lower cd_dp's `PLAN_RH` below 15 when comparing to MPCC/InvDyn (5× faster replan = unfair advantage).
5. **Don't `pip install cleandiffuser`** — its `numba<0.60` pin forces a numpy 1.x downgrade that breaks opencv/torch. `setup_env.sh` handles deps via leaf-package install.
6. **`<TS>` placeholder in paths**: training stamps a timestamp into the exp dir. Find newest with `ls -td $STABLEWM_HOME/latents/dp_cube_aw_image/cube_aw_chi_unet_ddpm_step50_h16_* | head -1`.

---

## PushT SV (single-view) — framework ready, SR not yet verified

Same pipeline pattern as cube AW. Dataset adapters + configs + submit scripts shipped; paper SR numbers TBD (requires user verification).

### Data

| H5 | Use |
|---|---|
| `pusht_expert_train.h5` (44GB, 18685 ep) | WM training base + cd_dp pixel train source |
| `pusht_dp_c0.5_success.h5` (881MB, 738 ep) | Slow expert demos — MPCC eval ref + latent diffusion target source |
| `pusht_dp_c1.0_success.h5` (1.7GB, 1671 ep) | Fast expert demos — alternative training source |

H5 schema (verified): `pixels (T, 224, 224, 3) uint8` + `action (T, 2) float32` + `proprio (T, 4)` + `state (T, 7)` + `ep_offset/ep_len`. Single view (cube has dual `pixels` + `wrist_pixels`).

### Train

```bash
# cd_dp pixel DP (pushT_expert_train, 18685 episodes — dataset load ~1.5h):
DATASET_PATH=$STABLEWM_HOME/pusht_expert_train.h5 \
PY=$PY CD_ROOT=$CD_ROOT LD_ROOT=$LD_ROOT \
sbatch $LD_ROOT/scripts/submit_train_dp_pusht_sv.sh

# Latent diffusion (requires pre-cached pushT latent H5 — see below):
sbatch --array=0,1,4,5%4 $LD_ROOT/scripts/submit_train_pusht_latent.sh   # all 4 ablations
```

### Prereq: cache pushT latent H5

The latent dataset class (`pusht_latent_dataset.py`) expects a pre-cached latent H5 (mirroring cube's `cube_multiview_diffusion_aw_c0.3.h5`). Default path:
```
$STABLEWM_HOME/pusht/latents/expert_slow_025_n200_centered_v2_park_latents_lewm.h5
```
**This file doesn't ship — run `scripts/cache_latents.py`** against `pusht_dp_c0.5_success.h5` (slow data) with a pushT JEPA WM ckpt (`pusht/lewm_object.ckpt`) before submitting latent train jobs. (cache_latents.py lives in the parent le-wm tree.)

### Eval

```bash
# cd_dp pixel eval (Hydra `--config-name pusht`):
DIFF_CKPT=<...>/dp_pusht_sv_image/<exp>/42/models/model_*.pt \
TRAIN_H5=$STABLEWM_HOME/pusht_expert_train.h5 \
POLICY=pusht/lewm \
sbatch $LD_ROOT/scripts/submit_eval_cd_dp_pusht_sv.sh

# Diffusion-MPCC eval (latent ckpt):
DIFF_CKPT=<...>/pusht_latent_velloss/<exp>/42/models/model_*.pt \
POLICY=pusht/lewm \
sbatch $LD_ROOT/scripts/submit_eval_diffusion_mpcc_pusht_sv.sh
```

### Files

```
latent_diffusion_cd/dataset/
  pusht_sv_image_dataset.py     # 224 → 96 resize, single `pixels` view, action_dim=2
  pusht_latent_dataset.py        # pre-cached latent H5 reader (mirrors cube)
configs/
  dp/pusht_sv/chi_unet/pusht_sv_image.yaml
  pusht_latent/{base,A_std,A_mm,C_std,C_mm}.yaml   # B_* skipped (broken for latent)
scripts/
  submit_train_dp_pusht_sv.sh
  submit_train_pusht_latent.sh
  submit_eval_cd_dp_pusht_sv.sh
  submit_eval_diffusion_mpcc_pusht_sv.sh
```

### PushT-specific gotchas

- `pusht_expert_train.h5` has 18685 episodes — initial zarr cache build is ~1.5h (vs cube's 30min). Consider `+max_episodes=N` override for smoke runs.
- pushT env (`swm/PushT-v1`) uses `history_size=3`, `frame_skip=1` — different from cube's `history_size=1`, `frame_skip=5`. Already handled by `config/eval/pusht.yaml`.
- pushT WM ckpt path: `pusht/lewm_object.ckpt` (vs cube's `pv_aw_cube_scratch_wm_training/lewm_pv_aw_cube_scratch_epoch_11`).
- Latent cache step: needed once before any latent diffusion training. SR ranking across A/B/C × std/mm may differ from cube (re-tune hyperparams).
