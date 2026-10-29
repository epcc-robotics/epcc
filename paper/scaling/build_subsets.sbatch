#!/usr/bin/env bash
# Build the nested, O/B/D-stratified subsets + common held-out val file of one
# aggregated corpus for the data-scaling sweep (scripts/data/build_scaling_subsets.py).
# The split is deterministic in (SRC, SEED, VAL_FRAC, DENOMS), so several jobs can
# write different parts in parallel:
#
#   F=findings/261001_scaling/scripts/build_subsets.sbatch
#   sbatch --export=ALL,TASK=plain,WRITE="val 256 64 16" $F
#   sbatch --export=ALL,TASK=plain,WRITE="4" $F
#   sbatch --export=ALL,TASK=plain,WRITE="1" $F
#SBATCH --job-name=scal_subsets
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPO=$REPO_ROOT
H=$STABLEWM_HOME
cd "$REPO"

TASK="${TASK:?set TASK=plain|slippery|cube}"
case "$TASK" in
  plain)    SRC="$H/aggcorpus/plain_main2.h5";    PREFIX=plain_main2 ;;
  slippery) SRC="$H/aggcorpus/slippery_main2.h5"; PREFIX=slippery_main2 ;;
  cube)     SRC="$H/aggcorpus/cube_main2.h5";     PREFIX=cube_main2 ;;
  *) echo "unknown TASK=$TASK"; exit 2 ;;
esac
OUT_DIR="$H/datasets/scaling/$TASK"
DENOMS="${DENOMS:-256 64 16 4 1}"
WRITE="${WRITE:-all}"

echo "TASK=$TASK SRC=$SRC OUT_DIR=$OUT_DIR DENOMS=[$DENOMS] WRITE=[$WRITE] JOB=$SLURM_JOB_ID host=$(hostname)"
python -u scripts/data/build_scaling_subsets.py \
    --src "$SRC" --out_dir "$OUT_DIR" --prefix "$PREFIX" \
    --denoms $DENOMS --val_frac "${VAL_FRAC:-0.01}" --seed "${SEED:-0}" \
    --write $WRITE ${VAL_CONTIG:+--val_contiguous $VAL_CONTIG --val_guard ${VAL_GUARD:-3}}
rc=$?
echo "=== DONE build_subsets rc=$rc ==="
exit $rc
