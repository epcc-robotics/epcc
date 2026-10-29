#!/bin/bash
#SBATCH --job-name=cd_dp_pusht_sv_skip
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# SkiP-relabeled PushT SV DP (SkiP baseline port to pushT).
# Mirrors submit_train_dp_cube_aw_skip.sh. PushTSVSkipImageDataset runs MSK on
# the AGENT-POSITION path and refits the action to an ABSOLUTE agent position
# [x, y]; skip-region targets leap to the next key-segment entrance.
# See configs/dp/pusht_sv/chi_unet/pusht_sv_image_skip.yaml.
#
# Prereqs: the repo environment (cleandiffuser is installed from third_party/).
#
# Submit (plain pushT — the config's own dataset):
#   sbatch submit_train_dp_pusht_sv_skip.sh
#
# Slippery variant — same recipe, different h5:
#   DATASET_PATH=<...>/slippery_pusht/data/expert_slow_050_n200_d035.h5 \
#   sbatch submit_train_dp_pusht_sv_skip.sh
#
# Optional env:
#   DATASET_PATH   override the h5 (default = the config's plain expert_slow_025)
#   GRADIENT_STEPS default 100000 (small = smoke)
#   MAX_EP         cap episodes (smoke); empty = all
#   STOP_FRAC      MSK velocity-stop keyframe fraction; empty = dataset default (0.25)
#   CUDA_MODULE    if set, `module load $CUDA_MODULE` first (e.g. cuda/12.9.1)
#
# Extra Hydra overrides pass straight through:
#   sbatch submit_train_dp_pusht_sv_skip.sh batch_size=128

set -e
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1

[ -n "${CUDA_MODULE:-}" ] && module load "$CUDA_MODULE"

PY="${PY:-python}"
LD_ROOT="${LD_ROOT:-$REPO_ROOT/latent_modules/diffusion_cd}"
REPO_ROOT="${REPO_ROOT:-$(cd "$LD_ROOT/../.." && pwd)}"
LATENT_DIFFUSION_DIR="${LATENT_DIFFUSION_DIR:-$REPO_ROOT/latent_modules/diffusion}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

GRADIENT_STEPS=${GRADIENT_STEPS:-100000}
MAX_EP=${MAX_EP:-}
STOP_FRAC=${STOP_FRAC:-}
DATASET_PATH=${DATASET_PATH:-}

EXTRA=""
[ -n "$DATASET_PATH" ] && EXTRA="$EXTRA dataset_path=$DATASET_PATH"
[ -n "$MAX_EP" ]       && EXTRA="$EXTRA ++dataset.max_episodes=$MAX_EP"
[ -n "$STOP_FRAC" ]    && EXTRA="$EXTRA ++dataset.stop_frac=$STOP_FRAC"

JOB_TAG=${SLURM_JOB_ID:-local_$$}

echo "=========================================================="
echo "[cd-dp pushT SV SkiP] ChiUNet1d + MultiImageObsCondition, MSK relabel"
echo "  LD_ROOT        = $LD_ROOT"
echo "  dataset_path   = ${DATASET_PATH:-<config default: plain expert_slow_025>}"
echo "  gradient_steps = $GRADIENT_STEPS"
echo "  max_episodes   = ${MAX_EP:-all}"
echo "  stop_frac      = ${STOP_FRAC:-<dataset default 0.25>}"
echo "  job_tag        = $JOB_TAG"
echo "  host           = $(hostname)"
echo "=========================================================="

cd "$LD_ROOT"
$PY -c "import cleandiffuser, latent_diffusion_cd; from latent_diffusion_cd.dataset.pusht_sv_skip_image_dataset import PushTSVSkipImageDataset; print('imports OK')"
$PY pipelines/dp_image.py \
    --config-path ../configs/dp/pusht_sv/chi_unet \
    --config-name pusht_sv_image_skip \
    gradient_steps="$GRADIENT_STEPS" \
    $EXTRA \
    "$@"

echo "[cd-dp pushT SV SkiP] done at $(date)"
