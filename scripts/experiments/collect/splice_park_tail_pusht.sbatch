#!/usr/bin/env bash
#SBATCH --job-name=splice_park
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
export SDL_VIDEODRIVER=dummy
SCRIPT=$REPO_ROOT/scripts/data/splice_park_tail_pusht.py
echo "=== splice: SRC=$SRC_H5 OUT=$OUT_H5 DAMPING=${DAMPING:-0} N_PARK=${N_PARK:-100} MAX_EP=${MAX_EP:-all} JOB=$SLURM_JOB_ID ==="
python "$SCRIPT"
echo "=== DONE rc=$? ==="
