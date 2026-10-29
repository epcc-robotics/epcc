#!/bin/bash
#SBATCH --job-name=cd_latent_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

# CleanDiffuser latent-diffusion ablations (input = frozen JEPA WM latents,
# output = action chunk).
# 7 cells: A_std A_mm B_std B_mm C_std C_mm B_mm_canonical.
#
# Prereqs: the repo environment (cleandiffuser is installed from third_party/).
#
# Submit examples:
#   sbatch --array=4         submit_train_cube_latent.sh    # C_std default
#   sbatch --array=0,2,4%3   submit_train_cube_latent.sh    # StandardScaler triple
#   sbatch --array=1,3,5%3   submit_train_cube_latent.sh    # MinMax triple
#   sbatch --array=6         submit_train_cube_latent.sh    # B_mm_canonical

set -e
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1

PY="${PY:-python}"
LD_ROOT="${LD_ROOT:-$REPO_ROOT/latent_modules/diffusion_cd}"


ABL_NAMES=(A_std A_mm B_std B_mm C_std C_mm B_mm_canonical)
IDX=${SLURM_ARRAY_TASK_ID:-4}
if (( IDX < 0 || IDX >= ${#ABL_NAMES[@]} )); then
    echo "[cd-latent] invalid array index $IDX" >&2
    exit 2
fi
ABL=${ABL_NAMES[$IDX]}

GRADIENT_STEPS=${GRADIENT_STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-2.0e-4}
SEED=${SEED:-42}

JOB_TAG=${SLURM_JOB_ID:-local_$$}

echo "=========================================================="
echo "[cd-latent train] cube AW latent diffusion"
echo "  array idx      = $IDX"
echo "  ablation       = $ABL"
echo "  LD_ROOT        = $LD_ROOT"
echo "  gradient_steps = $GRADIENT_STEPS"
echo "  batch_size     = $BATCH_SIZE"
echo "  lr             = $LR"
echo "  seed           = $SEED"
echo "  job_tag        = $JOB_TAG"
echo "  host           = $(hostname)"
echo "=========================================================="

cd "$LD_ROOT"
$PY pipelines/train_latent_velloss.py \
    --config-name "$ABL" \
    gradient_steps="$GRADIENT_STEPS" \
    batch_size="$BATCH_SIZE" \
    lr="$LR" \
    seed="$SEED" \
    "$@"

echo "[cd-latent train] $ABL done at $(date)"
