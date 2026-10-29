#!/usr/bin/env bash
# SAIL step 2/3 — train the reached-pose diffusion policy.
#
# Builds the label h5 first if it is missing, then trains a 6-D CleanDiffuser
# DP on it: [x, y, z, yaw, gripper_cmd, precision]. Architecture and
# hyperparameters are identical to the 5-D relative baseline — only the action
# target changes — so any difference is attributable to the target, not the model.
#
# Output lands in {beyond_demo}/{task}/sail/dp_reached/ (models/ +
# normalizer_stats.npz + .hydra/), which is what config/eval/.../
# {task}_sail_reached.yaml points at.
#
# Usage:
#   bash scripts/sail/train_reached_dp.sh              # full 100k-step run (~80 min on a 4090)
#   DRY_RUN=1 bash scripts/sail/train_reached_dp.sh    # print the plan, touch nothing
#   GRADIENT_STEPS=10000 bash scripts/sail/train_reached_dp.sh
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}
TASK=${TASK:-cube}

export CD_ROOT=${CD_ROOT:-$HOME/CleanDiffuser}

ASSETS=$STABLEWM_HOME/$TASK/sail
LABELS=${LABELS:-$ASSETS/awe/${TASK}_reached_precision_c03.h5}
GRADIENT_STEPS=${GRADIENT_STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1.0e-4}
SEED=${SEED:-42}
DRY_RUN=${DRY_RUN:-0}

echo "=================================================================="
echo " SAIL reached-pose DP training — task=$TASK"
echo "   labels         : $LABELS"
echo "   output         : $ASSETS/dp_reached/"
echo "   gradient_steps : $GRADIENT_STEPS   batch: $BATCH_SIZE   lr: $LR   seed: $SEED"
echo "=================================================================="

if [ ! -f "$LABELS" ]; then
  echo "[1/2] labels missing -> scripts/sail/make_labels.py"
  [ "$DRY_RUN" = "1" ] && echo "      (dry run: skipped)" \
    || $PY "$REPO/scripts/sail/make_labels.py" --out "$LABELS"
else
  echo "[1/2] labels present — skipping label build"
  [ "$DRY_RUN" = "1" ] || $PY - "$LABELS" <<'PYEOF'
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r") as f:
    a, p = f["action"], f["precision"][:]
    print(f"      action {a.shape} (expect (*, 6))   precision fraction {p.mean():.3f}")
PYEOF
fi

echo "[2/2] training"
if [ "$DRY_RUN" = "1" ]; then
  echo "      (dry run) would run: pipelines/dp_image.py dataset_path=$LABELS \\"
  echo "          shape_meta.action.shape=[6] action_dim=6 \\"
  echo "          save_path=$ASSETS gradient_steps=$GRADIENT_STEPS"
  exit 0
fi

# The launcher writes to {save_path}/dp_cube_aw_image/{exp_name}/{seed}; train
# into a staging dir, then flatten to the contract layout.
STAGE=$ASSETS/_train_staging
EXP=reached_precision_6d
DATASET_PATH=$LABELS PY=$PY CD_ROOT=$CD_ROOT LD_ROOT=$LD_ROOT \
  bash "$LD_ROOT/scripts/submit_train_dp_cube_aw.sh" \
    "shape_meta.action.shape=[6]" action_dim=6 \
    save_path="$STAGE" exp_name="$EXP" \
    gradient_steps="$GRADIENT_STEPS" batch_size="$BATCH_SIZE" lr="$LR" seed="$SEED"

RUN_DIR=$STAGE/dp_cube_aw_image/$EXP/$SEED
if [ -d "$ASSETS/dp_reached" ]; then
  BACKUP=$ASSETS/dp_reached.prev.$(date +%Y%m%d_%H%M%S)
  echo "  existing dp_reached/ -> $BACKUP"
  mv "$ASSETS/dp_reached" "$BACKUP"
fi
mv "$RUN_DIR" "$ASSETS/dp_reached"
rm -rf "$STAGE"

echo
echo "done. checkpoints:"
ls "$ASSETS/dp_reached/models/" | tail -3
echo "point config/eval/baseline_corl_v2/${TASK}_sail_reached.yaml at the ckpt you want,"
echo "then: bash scripts/sail/eval_sail.sh"
