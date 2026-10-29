#!/bin/bash
#SBATCH --job-name=cd_dp_pusht_sv
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

# CleanDiffuser DP recipe — pushT SV pixel input.
# Mirrors cube_aw recipe; pushT single-view (1 rgb key) + action_dim=2.
#
# Submit:
#   DATASET_PATH=/path/to/pusht_expert_train.h5 \
#   sbatch submit_train_dp_pusht_sv.sh

set -e
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1

PY="${PY:-python}"
LD_ROOT="${LD_ROOT:-$REPO_ROOT/latent_modules/diffusion_cd}"


DATASET_PATH=${DATASET_PATH:?'set DATASET_PATH to your pusht_expert_train.h5'}
GRADIENT_STEPS=${GRADIENT_STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1.0e-4}
SEED=${SEED:-42}

JOB_TAG=${SLURM_JOB_ID:-local_$$}

echo "=========================================================="
echo "[cd-dp pushT SV] CleanDiffuser ChiUNet1d + MultiImageObsCondition"
echo "  dataset_path   = $DATASET_PATH"
echo "  gradient_steps = $GRADIENT_STEPS"
echo "  batch_size     = $BATCH_SIZE"
echo "  lr             = $LR"
echo "  seed           = $SEED"
echo "  job_tag        = $JOB_TAG"
echo "  host           = $(hostname)"
echo "=========================================================="

cd "$LD_ROOT"
$PY pipelines/dp_image.py \
    --config-path ../configs/dp/pusht_sv/chi_unet \
    --config-name pusht_sv_image \
    "dataset_path=$DATASET_PATH" \
    gradient_steps="$GRADIENT_STEPS" \
    batch_size="$BATCH_SIZE" \
    lr="$LR" \
    seed="$SEED" \
    "$@"

echo "[cd-dp pushT SV] done at $(date)"
