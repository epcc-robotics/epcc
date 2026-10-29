#!/bin/bash
#SBATCH --job-name=cd_pusht_latent
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

# CleanDiffuser latent-diffusion ablations on pushT SV cached latents.
# Prereq: pusht latent H5 must exist — run scripts/cache_latents.py first
# against pusht_dp_c0.5_success.h5 (or c=1.0) using a pushT JEPA WM ckpt.
#
# Array indices: 0=A_std, 1=A_mm, 4=C_std, 5=C_mm (B variants skipped — broken)
#
# Submit examples:
#   sbatch --array=0,1,4,5%4 submit_train_pusht_latent.sh   # all 4 best variants
#   sbatch --array=4         submit_train_pusht_latent.sh   # C_std default

set -e
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1

PY="${PY:-python}"
LD_ROOT="${LD_ROOT:-$REPO_ROOT/latent_modules/diffusion_cd}"


ABL_NAMES=(A_std A_mm B_std B_mm C_std C_mm)
IDX=${SLURM_ARRAY_TASK_ID:-4}
if (( IDX < 0 || IDX >= ${#ABL_NAMES[@]} )); then
    echo "[cd-pusht-latent] invalid array index $IDX" >&2
    exit 2
fi
ABL=${ABL_NAMES[$IDX]}

GRADIENT_STEPS=${GRADIENT_STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-2.0e-4}
SEED=${SEED:-42}

JOB_TAG=${SLURM_JOB_ID:-local_$$}

echo "=========================================================="
echo "[cd-pusht-latent train] pushT SV latent diffusion"
echo "  array idx      = $IDX"
echo "  ablation       = $ABL"
echo "  gradient_steps = $GRADIENT_STEPS"
echo "  batch_size     = $BATCH_SIZE"
echo "  lr             = $LR"
echo "  seed           = $SEED"
echo "  job_tag        = $JOB_TAG"
echo "  host           = $(hostname)"
echo "=========================================================="

cd "$LD_ROOT"
$PY pipelines/train_latent_velloss.py \
    --config-path ../configs/pusht_latent \
    --config-name "$ABL" \
    gradient_steps="$GRADIENT_STEPS" \
    batch_size="$BATCH_SIZE" \
    lr="$LR" \
    seed="$SEED" \
    "$@"

echo "[cd-pusht-latent train] $ABL done at $(date)"
