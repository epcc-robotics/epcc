#!/usr/bin/env bash
# Chain step after one scaling WM training job (submitted by launch_sweep.sh with
# --dependency=afterok:<train job>):
#   1. pick the held-out-val-minimum epoch and register it as checkpoints/scaling/<task>_f<D>
#   2. submit the EPCC eval of that WM (paper cell paper/scaling/<task>_f<D>)
#   3. submit the scoring of that eval, dependent on it
#SBATCH --job-name=scal_post
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
cd "$REPO_ROOT" || exit 1
TASK="${TASK:?}"; DENOM="${DENOM:?}"; TRAIN_JID="${TRAIN_JID:?}"
NAME="scal_${TASK}_f${DENOM}"
LOG="logs/${NAME}_${TRAIN_JID}.out"
RUN_DIR="$STABLEWM_HOME/scaling_runs/${NAME}_${TRAIN_JID}"
case "$TASK" in plain) CT=pusht ;; slippery) CT=slippery_pusht ;; cube) CT=cube ;; *) echo "unknown task $TASK"; exit 2 ;; esac

python3 paper/scaling/select_ckpt.py --log "$LOG" --run_dir "$RUN_DIR" \
    --name "$NAME" --group "scaling/${TASK}_f${DENOM}" --link || exit 1

CELL="${CT}_f${DENOM}"
EVAL_JID=$(SLURM_TIME=04:00:00 LOGS_SUBDIR=scaling paper/run_cell.sh --slurm "paper/scaling/$CELL" "$CELL") || { echo "eval not submitted"; exit 1; }
RES=${RESULTS_DIR:-results}; case "$RES" in /*) ;; *) RES="$REPO_ROOT/$RES" ;; esac
SJ=$(scripts/submit.sh --parsable --dependency=afterok:"$EVAL_JID" --job-name="score_$CELL" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",RUNS="$RES/scaling/$CELL",OUT="$RES/scaling/$CELL/scores.json" \
    paper/common/score_runs.sbatch)
echo "post_train $NAME: registered, eval job=$EVAL_JID, score job=$SJ"
