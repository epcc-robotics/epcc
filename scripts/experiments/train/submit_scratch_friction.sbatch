#!/usr/bin/env bash
# Friction-sweep WM, one damping value per job. FROM SCRATCH with the encoder
# trainable, so each value gets its own latent space and therefore needs its own
# latent cache and its own DTG downstream.
#
# Recipe is lewm_slippery_pusht_v2scratch verbatim — same ViT-tiny / 192-d /
# history 3 / frameskip 5 / SIGReg 0.09 / AdamW 5e-5 as the champion. Only the
# corpus changes, which is the whole point: anything else moving would confound
# the friction variable.
#
# `trainer.precision=32` overrides the config's bf16. Every from-scratch and
# finetune launcher in this repo does the same (7/7), so this follows the house
# convention rather than the config default.
#
# The h5 is staged to node-local /tmp first (as submit_scratch_aggM2.sbatch does)
# because the training loader hammers random offsets and the shared filesystem is
# the bottleneck otherwise. Friction corpora are 1.8-2.8 GB so the copy is cheap.
#
#   sbatch --export=ALL,D=d030 scripts/experiments/train/submit_scratch_friction.sbatch
#
#SBATCH --job-name=lewm_fric
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
D=${D:?set D via --export, e.g. D=d030}

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPRO=$REPO_ROOT

NAME="friction_${D}_excised"
SRC="$STABLEWM_HOME/slippery_pusht/data/${NAME}.h5"
[ -e "$SRC" ] || { echo "MISSING: $SRC"; exit 1; }

LOCAL_BASE="/tmp/$USER/stable-wm-${SLURM_JOB_ID}"
mkdir -p "$LOCAL_BASE/datasets"
echo "staging $SRC -> $LOCAL_BASE/datasets/ ..."
cp -L "$SRC" "$LOCAL_BASE/datasets/" || exit 1

SUBDIR="fric${D}_${SLURM_JOB_ID}"
MODEL="lewm_fric_${D}"
cd "$REPRO" || exit 1
echo "=== friction WM  D=$D  subdir=$SUBDIR  JOB=$SLURM_JOB_ID ==="

python train.py --config-name lewm_slippery_pusht_v2scratch \
  trainer.precision=32 \
  data.dataset.name="$NAME" \
  data.dataset.cache_dir="$LOCAL_BASE" \
  output_model_name="$MODEL" \
  subdir="$SUBDIR" \
  wandb.config.id="$SUBDIR"
rc=$?
rm -rf "$LOCAL_BASE"

# Register the trained encoder under the alias the eval resolver expects.
# Training writes <subdir>/<model>_epoch_N_object.ckpt, which is not the
# <group>_object.ckpt name a cell can point at. Same pattern as
# checkpoints/agg_slipM2scr_object.ckpt and checkpoints/v1scr20_object.ckpt.
# Take the HIGHEST epoch present rather than assuming 20, so a job that stopped
# early registers what it actually produced instead of a dangling link.
if [ "$rc" = "0" ]; then
    LAST=$(ls -1 "$STABLEWM_HOME/$SUBDIR"/${MODEL}_epoch_*_object.ckpt 2>/dev/null \
           | sed 's/.*_epoch_\([0-9]*\)_object.ckpt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
    if [ -n "$LAST" ]; then
        ln -sfn "../$SUBDIR/$(basename "$LAST")" \
                "$STABLEWM_HOME/checkpoints/fric${D}_object.ckpt"
        echo "registered: checkpoints/fric${D}_object.ckpt -> $(basename "$LAST")"
    else
        echo "WARN: no epoch checkpoint found in $STABLEWM_HOME/$SUBDIR"
    fi
fi
echo "=== DONE D=$D rc=$rc ==="
