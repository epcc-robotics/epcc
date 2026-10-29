#!/usr/bin/env bash
# Chain step after one f4cold training job (submit with --dependency=afterok:<train job>):
#   1. pick the lowest validate/pred_loss_epoch and register it as f4cold/<task>_<POINT>
#   2. submit the EPCC eval of that world model (paper cell paper/fig4/<task>_<POINT>)
#
#   sbatch --dependency=afterok:<TRAIN_JID> --job-name=f4cold_post \
#          --export=ALL,TASK=plain,POINT=D05,TRAIN_JID=<TRAIN_JID> paper/fig4/cold/post_f4cold.sbatch
#SBATCH --job-name=f4cold_post
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
cd "$REPO_ROOT" || exit 1
TASK="${TASK:?}"; POINT="${POINT:?}"; TRAIN_JID="${TRAIN_JID:?}"
NAME="f4cold_${TASK}_${POINT}"
LOG="logs/${NAME}_${TRAIN_JID}.out"
RUN_DIR="$STABLEWM_HOME/fig4cold_runs/${NAME}_${TRAIN_JID}"
# resumed run: PREV_LOGS="<older job stdout> ..." (in order), RUN_DIR_OVERRIDE=<the original run dir>
LOGARGS=()
for lg in ${PREV_LOGS:-}; do LOGARGS+=(--log "$lg"); done
LOGARGS+=(--log "$LOG")
RUN_DIR="${RUN_DIR_OVERRIDE:-$RUN_DIR}"

python3 paper/fig4/cold/select_ckpt.py "${LOGARGS[@]}" --run_dir "$RUN_DIR" \
    --name "$NAME" --group "f4cold/${TASK}_${POINT}" --link || exit 1

case "$TASK" in plain) CT=pusht ;; slippery) CT=slippery_pusht ;; *) echo "unknown task $TASK"; exit 2 ;; esac
CELL="${CT}_${POINT}"
EVAL_JID=$(SLURM_TIME=04:00:00 LOGS_SUBDIR=fig4 paper/run_cell.sh --slurm "paper/fig4/$CELL" "$CELL") || { echo "eval not submitted"; exit 1; }
echo "post $NAME: registered checkpoints/f4cold/${TASK}_${POINT}, eval job=$EVAL_JID"
echo "=== DONE post rc=0 ==="
