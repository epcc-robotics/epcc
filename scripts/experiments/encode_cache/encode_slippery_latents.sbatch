#!/usr/bin/env bash
# Encode slippery PushT expert demos -> LeWM latents (pool=lewm, dim 192).
# Params via --export: CKPT (WM object.ckpt), SRC (raw h5), OUT (latent h5).
#SBATCH --job-name=enc_slip
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPRO=$REPO_ROOT
cd "$REPRO"

CKPT=${CKPT:?set CKPT}
SRC=${SRC:?set SRC}
OUT=${OUT:?set OUT}
mkdir -p "$(dirname "$OUT")"

FP32=${FP32:-0}
FP32FLAG=""; [ "$FP32" = "1" ] && FP32FLAG="--fp32"
echo "=== ENCODE  ckpt=$CKPT"
echo "           src =$SRC  fp32=$FP32"
echo "           out =$OUT"
python scripts/cache_latents.py --dataset pusht \
  --ckpt "$CKPT" --source "$SRC" --output "$OUT" \
  --batch_size 256 --pool lewm $FP32FLAG
rc=$?
echo "=== encode rc=$rc ==="
[ $rc -eq 0 ] && h5ls "$OUT" 2>/dev/null
echo "=== DONE ==="
