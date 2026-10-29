#!/usr/bin/env bash
# Data-scaling sweep, Cube: one per-view (agent+wrist) WM per nested subset of agg_cube_main2.
# Same recipe as Table I's agg_cubeM2 (scripts/experiments/train/train_perview_cube_demoonly.sbatch
# with DATASET=agg_cube_main2, MAX_EPOCHS=7): encoder+projector warm-started from the cube
# donor and FROZEN, predictor/action_encoder/pred_proj/grasp head from scratch (cold, as in
# every cube WM), lr_groups encoder=action_encoder=1.0, proprio token off, fp32, batch 128.
# Added for the sweep (train_perview_WM_cube.py, default-off options):
#   +steps_per_epoch       fixed-length epochs (val + object dump every SPE steps)
#   +data.val_dataset_name common held-out val file
#   +lr_cosine_tmax=7      keeps Table I's LR dynamics: spt steps the cosine every optimizer
#                          step, so T_max = max_epochs = 7 gave a 14-step LR cycle there
#
#   S=findings/261001_scaling/scripts/train_wm_scaling_cube.sbatch
#   sbatch --job-name=scal_cube_f256 --export=ALL,DENOM=256,SPE=2000,MAX_EPOCHS=50 $S
#   sbatch --job-name=scal_smoke_cube_f256 --time=01:00:00 --export=ALL,DENOM=256,SMOKE=1 $S
#SBATCH --job-name=wm_scaling_cube
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=90G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:TERM@300
set -uo pipefail
NVIDIA_LIB_PATHS=""
ENV_SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")  # site-packages of the active env (pip nvidia libs)
for d in "$ENV_SITE"/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
done
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:-}"
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPO=$REPO_ROOT
H=$STABLEWM_HOME

DENOM="${DENOM:?set DENOM (256|64|16|4|1)}"
SPE="${SPE:-2000}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
LR_TMAX="${LR_TMAX:-7}"
WARM_CKPT="$H/checkpoints/cube_wm/lewm_pv_aw_cube_scratch_epoch_11_object.ckpt"
SMOKE_ARGS=(); SMOKE_TAG=""; WANDB_ON=true
if [ "${SMOKE:-0}" = "1" ]; then
    SPE=20; MAX_EPOCHS=2; SMOKE_TAG="_smoke"; WANDB_ON=false
    SMOKE_ARGS=(+trainer.limit_val_batches=5)
    echo "[SMOKE] 2 epochs x 20 steps, 5 val batches, wandb off"
fi

DS_REL="scaling/cube/cube_main2_f${DENOM}"
VAL_REL="scaling/cube/cube_main2_val"
OUT_NAME="${OUT_NAME:-scal_cube_f${DENOM}${SMOKE_TAG}}"
SRC="$H/datasets/${DS_REL}.h5"
VAL="$H/datasets/${VAL_REL}.h5"
for f in "$SRC" "$VAL" "$WARM_CKPT"; do
    [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done

LOCAL_BASE="/tmp/$USER/stable-wm-${SLURM_JOB_ID:-local_$$}"
LOCAL_DATA="$LOCAL_BASE/datasets/scaling/cube"
mkdir -p "$LOCAL_DATA"
trap 'rm -rf "$LOCAL_BASE"' EXIT
t0=$(date +%s)
cp "$SRC" "$VAL" "$LOCAL_DATA/" || exit 1
echo "[staging] $(du -shL "$SRC" | cut -f1) + $(du -shL "$VAL" | cut -f1) -> $LOCAL_DATA in $(( $(date +%s) - t0 ))s"

SUBDIR="scaling_runs/${OUT_NAME}_${SLURM_JOB_ID:-local}"
cd "$REPO"
echo "=========================================================="
echo "[scaling-cube] DENOM=1/$DENOM  dataset=$DS_REL  val=$VAL_REL"
echo "  encoder   = $WARM_CKPT (encoder+projector, frozen) ; dynamics = fresh init"
echo "  schedule  = $MAX_EPOCHS epochs x $SPE steps = $(( MAX_EPOCHS * SPE )) steps ; LR cosine T_max=$LR_TMAX steps"
echo "  run dir   = $H/$SUBDIR   job=${SLURM_JOB_ID:-local}  host=$(hostname)"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u train_perview_WM_cube.py \
    "${SMOKE_ARGS[@]}" \
    subdir="$SUBDIR" \
    output_model_name="$OUT_NAME" \
    data.dataset.name="$DS_REL" \
    +data.dataset.cache_dir="$LOCAL_BASE" \
    +data.val_dataset_name="$VAL_REL" \
    +steps_per_epoch="$SPE" \
    +lr_cosine_tmax="$LR_TMAX" \
    trainer.precision=32 \
    trainer.max_epochs="$MAX_EPOCHS" \
    tokens.proprio.enabled=false \
    warmstart.ckpt_path="$WARM_CKPT" \
    +freeze_encoder=true \
    warmstart.modules.encoder=true \
    warmstart.modules.projector=true \
    warmstart.modules.action_encoder=false \
    warmstart.modules.pred_proj=false \
    warmstart.modules.predictor=false \
    lr_groups.encoder=1.0 \
    lr_groups.action_encoder=1.0 \
    wandb.enabled="$WANDB_ON" \
    wandb.config.project=lewm-scaling \
    wandb.config.name="$OUT_NAME" \
    wandb.config.id="${OUT_NAME}_${SLURM_JOB_ID:-local}" \
    wandb.config.resume=allow
rc=$?
echo "=== DONE train rc=$rc run_dir=$H/$SUBDIR ==="
exit $rc
