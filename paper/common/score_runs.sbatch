#!/usr/bin/env bash
# Score campaign run dirs on a CPU node (numpy). Usage:
#   sbatch --export=ALL,RUNS="<abs run dir> <abs run dir> ...",OUT=<abs json> \
#          findings/260907_final_dtg_s1_warm/scripts/score_runs.sbatch
#SBATCH --job-name=score_runs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
unset LD_PRELOAD
RUNS=${RUNS:?set RUNS}; OUT=${OUT:?set OUT}
python -u "$REPO_ROOT/paper/common/score_runs.py" --out "$OUT" $RUNS
echo "=== DONE score_runs rc=$? ==="
