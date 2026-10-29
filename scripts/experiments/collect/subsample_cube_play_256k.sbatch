#!/usr/bin/env bash
# EXPERIMENT 6 control cell, step 1 -- cut the cube PLAY corpus down to the demo
# frame budget so "demo vs play" is not confounded with "less data vs more".
#
#   play full   2,010,000 frames / 10,000 eps   <- champion WM trained here
#   demo         256,000 frames /    800 eps
#   play-sub     256,000 frames /  ~1,274 eps   <- built here
#
# Whole episodes, seeded selection (never a prefix -- see the note in
# subsample_episodes_h5.py). Output keeps the source's 37-key schema unchanged,
# so the training launcher runs on it with DATASET= and no other edit.
#
#SBATCH --job-name=sub_cube_play
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPO=$REPO_ROOT
MV="$STABLEWM_HOME/cube/collect"
cd "$REPO" || exit 1

echo "=== subsample cube play -> 256k frames  JOB=$SLURM_JOB_ID ==="
df -h "$MV" | tail -1

python -u scripts/data/subsample_episodes_h5.py \
    --src "$MV/cube_multiview_WM.h5" \
    --out "$MV/cube_multiview_WM_sub256k.h5" \
    --target_frames 256000 \
    --seed 42
rc=$?
echo "=== rc=$rc ==="
[ $rc -eq 0 ] && ls -lh "$MV/cube_multiview_WM_sub256k.h5"
exit $rc
