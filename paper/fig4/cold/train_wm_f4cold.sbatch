#!/usr/bin/env bash
# Fig. 4 world models for PushT and Slippery-PushT (the version reported in the paper).
#
# Base config, donor checkpoint, frozen encoder+projector, lr 5e-5, batch 128/256, seed 3072,
# default 0.9/0.1 window split, +reinit_dynamics=true (predictor / action_encoder / pred_proj
# start from fresh initialization, as for the Cube Fig. 4 models) and 100 epochs.
# Checkpoint = lowest validate/pred_loss_epoch (select_ckpt.py via post_f4cold.sbatch), as for Cube.
#
#   scripts/submit.sh --job-name=f4cold_plain_D05 --export=ALL,TASK=plain,POINT=D05 paper/fig4/cold/train_wm_f4cold.sbatch
#   then paper/fig4/cold/post_f4cold.sbatch (afterok) selects the checkpoint and submits the eval.
#SBATCH --job-name=f4cold
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --no-requeue          # a requeue truncates the .out and would shift select_ckpt.py's table->epoch mapping
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:TERM@300
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPO=$REPO_ROOT
H=$STABLEWM_HOME

TASK="${TASK:?set TASK=plain|slippery}"
POINT="${POINT:?set POINT=DEMO|CTRL|D05|D15|D25}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
SMOKE_ARGS=()
SMOKE_TAG=""
WANDB_ON=true
if [ "${SMOKE:-0}" = "1" ]; then
    MAX_EPOCHS=2; SMOKE_TAG="_smoke"; WANDB_ON=false
    SMOKE_ARGS=(+trainer.limit_train_batches=5 +trainer.limit_val_batches=2)
    echo "[SMOKE] 2 epochs x 5 train batches, 2 val batches, wandb off"
fi

# Training corpora of the published Fig. 4 models (their config.yaml, checked 2026-10-22).
case "$TASK:$POINT" in
  plain:DEMO)    DS_REL=expert_slow_025_n200_centered_v2_park ;;
  plain:CTRL)    DS_REL=pusht_expert_train_sub83k ;;
  plain:D05)     DS_REL=agg_plain_D_c0.5 ;;
  plain:D15)     DS_REL=agg_plain_D_c1.5 ;;
  plain:D25)     DS_REL=agg_plain_D_c2.5 ;;
  slippery:DEMO) DS_REL=slippery_pusht/data/expert_slow_050_n200_d035_park ;;
  slippery:CTRL) DS_REL=slippery_pusht/data/slippery_pusht_wm_corpus_v1_sub101k ;;
  slippery:D05)  DS_REL=agg_slippery_D_c0.5 ;;
  slippery:D15)  DS_REL=agg_slippery_D_c1.5 ;;
  slippery:D25)  DS_REL=agg_slippery_D_c2.5 ;;
  *) echo "unknown TASK:POINT=$TASK:$POINT"; exit 2 ;;
esac

case "$TASK" in
  plain)
    CFG=lewm                              # declares none of these keys -> append form
    FT_CKPT="$H/checkpoints/pusht_wm/lewm_object.ckpt"
    BASE=(+finetune_from="$FT_CKPT" +freeze_encoder=true
          +data.dataset.cache_dir="__LOCAL_BASE__")
    ;;
  slippery)
    CFG=lewm_slippery_pusht_v2ft          # declares finetune_from/freeze_encoder/cache_dir
    FT_CKPT="$H/checkpoints/slippery_pusht/lewm/lewm_object.ckpt"
    BASE=(finetune_from="$FT_CKPT" freeze_encoder=true
          data.dataset.cache_dir="__LOCAL_BASE__")
    ;;
esac

OUT_NAME="f4cold_${TASK}_${POINT}${SMOKE_TAG}"
SRC="$H/datasets/${DS_REL}.h5"
for f in "$SRC" "$FT_CKPT"; do
    [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done

# stage on node-local /tmp (Lustre random reads are the bottleneck)
LOCAL_BASE="/tmp/$USER/stable-wm-${SLURM_JOB_ID:-local_$$}"
LOCAL_DATA="$LOCAL_BASE/datasets/$(dirname "$DS_REL")"
mkdir -p "$LOCAL_DATA"
trap 'rm -rf "$LOCAL_BASE"' EXIT          # before cp: a failed copy must not leave GBs in /tmp
t0=$(date +%s)
cp "$SRC" "$LOCAL_DATA/" || exit 1
echo "[staging] $(du -shL "$SRC" | cut -f1) -> $LOCAL_DATA in $(( $(date +%s) - t0 ))s"
BASE=("${BASE[@]/__LOCAL_BASE__/$LOCAL_BASE}")

# run dir = $STABLEWM_HOME/<subdir> (train.py: swm get_cache_dir() / subdir)
SUBDIR="fig4cold_runs/${OUT_NAME}_${SLURM_JOB_ID:-local}"
# RESUME_SUBDIR=fig4cold_runs/<name>_<old jid>: continue a run that died, from its <name>_weights.ckpt
# (spt.Manager resumes trainer.fit from it: weights, optimizer, LR schedule, epoch counter).
if [ -n "${RESUME_SUBDIR:-}" ]; then
    [ -f "$H/$RESUME_SUBDIR/${OUT_NAME}_weights.ckpt" ] || { echo "RESUME: no $H/$RESUME_SUBDIR/${OUT_NAME}_weights.ckpt"; exit 1; }
    SUBDIR="$RESUME_SUBDIR"
    echo "[resume] continuing $H/$SUBDIR from ${OUT_NAME}_weights.ckpt"
fi

cd "$REPO"
echo "=========================================================="
echo "[f4cold-$TASK-$POINT] dataset=$DS_REL"
echo "  encoder   = $FT_CKPT (frozen) ; dynamics = fresh init"
echo "  epochs    = $MAX_EPOCHS"
echo "  run dir   = $H/$SUBDIR   job=${SLURM_JOB_ID:-local}  host=$(hostname)"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u train.py --config-name "$CFG" \
    "${BASE[@]}" "${SMOKE_ARGS[@]}" \
    +reinit_dynamics=true \
    trainer.precision=32 \
    trainer.max_epochs="$MAX_EPOCHS" \
    data.dataset.name="$DS_REL" \
    output_model_name="$OUT_NAME" \
    subdir="$SUBDIR" \
    wandb.enabled="$WANDB_ON" \
    wandb.config.project=lewm-fig4-cold \
    wandb.config.name="$OUT_NAME" \
    wandb.config.id="${OUT_NAME}_${SLURM_JOB_ID:-local}" \
    wandb.config.resume=allow
rc=$?
echo "=== DONE train rc=$rc run_dir=$H/$SUBDIR ==="
exit $rc
