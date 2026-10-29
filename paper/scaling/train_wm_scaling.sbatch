#!/usr/bin/env bash
# Data-scaling sweep: one world model per nested subset of agg_<task>_main2.
#   encoder+projector : task donor, FROZEN (keeps the latent space of the DTG and
#                       warm-ring caches, so they stay valid for every size)
#   predictor / action_encoder / pred_proj : fresh init (+reinit_dynamics=true),
#                       so no size inherits dynamics learned from the donor corpus
#   fixed-length epochs (+steps_per_epoch) and one common held-out val file
#   (+data.val_dataset_name) for checkpoint selection across sizes.
#
#   S=findings/261001_scaling/scripts/train_wm_scaling.sbatch
#   sbatch --job-name=scal_plain_f256 --export=ALL,TASK=plain,DENOM=256,MAX_EPOCHS=50 $S
#   sbatch --job-name=scal_smoke --export=ALL,TASK=plain,DENOM=256,SMOKE=1 $S
#SBATCH --job-name=wm_scaling
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=90G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:TERM@300
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPO=$REPO_ROOT
H=$STABLEWM_HOME

TASK="${TASK:?set TASK=plain|slippery}"
DENOM="${DENOM:?set DENOM (256|64|16|4|1)}"
SPE="${SPE:-2000}"                        # optimizer steps per epoch
MAX_EPOCHS="${MAX_EPOCHS:-50}"            # total steps = SPE * MAX_EPOCHS
SMOKE_ARGS=()
SMOKE_TAG=""
WANDB_ON=true
if [ "${SMOKE:-0}" = "1" ]; then
    SPE=20; MAX_EPOCHS=2; SMOKE_TAG="_smoke"; WANDB_ON=false
    SMOKE_ARGS=(+trainer.limit_val_batches=5)
    echo "[SMOKE] 2 epochs x 20 steps, 5 val batches, wandb off"
fi

case "$TASK" in
  plain)
    CFG=lewm
    FT_CKPT="$H/checkpoints/pusht_wm/lewm_object.ckpt"
    PREFIX=plain_main2
    # lewm.yaml declares none of these keys -> append form
    BASE=(+finetune_from="$FT_CKPT" +freeze_encoder=true
          +data.dataset.cache_dir="__LOCAL_BASE__")
    ;;
  slippery)
    CFG=lewm_slippery_pusht_v2ft          # declares finetune_from/freeze_encoder/cache_dir
    FT_CKPT="$H/checkpoints/slippery_pusht/lewm/lewm_object.ckpt"
    PREFIX=slippery_main2
    BASE=(finetune_from="$FT_CKPT" freeze_encoder=true
          data.dataset.cache_dir="__LOCAL_BASE__")
    ;;
  *) echo "unknown TASK=$TASK (want plain|slippery; cube uses the per-view trainer)"; exit 2 ;;
esac

DS_REL="scaling/$TASK/${PREFIX}_f${DENOM}"
VAL_REL="scaling/$TASK/${PREFIX}_val"
OUT_NAME="${OUT_NAME:-scal_${TASK}_f${DENOM}${SMOKE_TAG}}"
SRC="$H/datasets/${DS_REL}.h5"
VAL="$H/datasets/${VAL_REL}.h5"
for f in "$SRC" "$VAL" "$FT_CKPT"; do
    [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done

# stage both files on node-local /tmp (Lustre random reads are the bottleneck)
LOCAL_BASE="/tmp/$USER/stable-wm-${SLURM_JOB_ID:-local_$$}"
LOCAL_DATA="$LOCAL_BASE/datasets/scaling/$TASK"
mkdir -p "$LOCAL_DATA"
trap 'rm -rf "$LOCAL_BASE"' EXIT          # before cp: a failed copy must not leave GBs in /tmp
t0=$(date +%s)
cp "$SRC" "$VAL" "$LOCAL_DATA/" || exit 1
echo "[staging] $(du -shL "$SRC" | cut -f1) + $(du -shL "$VAL" | cut -f1) -> $LOCAL_DATA in $(( $(date +%s) - t0 ))s"
BASE=("${BASE[@]/__LOCAL_BASE__/$LOCAL_BASE}")

# run dir = $STABLEWM_HOME/<subdir> (train.py: swm get_cache_dir() / subdir)
SUBDIR="scaling_runs/${OUT_NAME}_${SLURM_JOB_ID:-local}"

cd "$REPO"
echo "=========================================================="
echo "[scaling-$TASK] DENOM=1/$DENOM  dataset=$DS_REL  val=$VAL_REL"
echo "  encoder   = $FT_CKPT (frozen) ; dynamics = fresh init"
echo "  schedule  = $MAX_EPOCHS epochs x $SPE steps = $(( MAX_EPOCHS * SPE )) steps"
echo "  run dir   = $H/$SUBDIR   job=${SLURM_JOB_ID:-local}  host=$(hostname)"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u train.py --config-name "$CFG" \
    "${BASE[@]}" "${SMOKE_ARGS[@]}" \
    +reinit_dynamics=true \
    +steps_per_epoch="$SPE" \
    +data.val_dataset_name="$VAL_REL" \
    trainer.precision=32 \
    trainer.max_epochs="$MAX_EPOCHS" \
    data.dataset.name="$DS_REL" \
    output_model_name="$OUT_NAME" \
    subdir="$SUBDIR" \
    wandb.enabled="$WANDB_ON" \
    wandb.config.project=lewm-scaling \
    wandb.config.name="$OUT_NAME" \
    wandb.config.id="${OUT_NAME}_${SLURM_JOB_ID:-local}" \
    wandb.config.resume=allow
rc=$?
echo "=== DONE train rc=$rc run_dir=$H/$SUBDIR ==="
exit $rc
