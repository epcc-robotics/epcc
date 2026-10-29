#!/usr/bin/env bash
# SAIL step 2/3 for cube — reached-pose 6-D DP.
#
# scripts/sail/train_reached_dp.sh is the original driver; this wrapper supplies the
# environment it expects (repo prologue, STABLEWM_HOME) and calls it. It in turn shells out to
# latent_modules/diffusion_cd/scripts/submit_train_dp_cube_aw.sh with
# `shape_meta.action.shape=[6] action_dim=6`, then flattens the staged run into
# the asset contract at $STABLEWM_HOME/cube/sail/dp_reached/.
#
# LABELS defaults to the file scripts/sail/make_labels.py writes. Chain this
# after the label job with --dependency=afterok so a failed label build cannot
# silently train on a stale or partial h5.
#
#SBATCH --job-name=dp_reached_cube
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
unset LD_PRELOAD
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPO=$REPO_ROOT

export PY=python
export TASK=cube
ENV_SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")  # site-packages of the active env (pip nvidia libs)
NV=""; for d in "$ENV_SITE"/nvidia/*/lib; do [ -d "$d" ] && NV="${d}:${NV}"; done
export LD_LIBRARY_PATH="${NV}${LD_LIBRARY_PATH:-}"

export LABELS=${LABELS:-$STABLEWM_HOME/cube/sail/awe/cube_reached_precision_c03.h5}
export GRADIENT_STEPS=${GRADIENT_STEPS:-100000}
[ "${SMOKE:-0}" = "1" ] && export GRADIENT_STEPS=200

if [ ! -f "$LABELS" ]; then
  echo "labels missing: $LABELS"
  echo "  (run scripts/sail/make_labels_all.sbatch with TASKS=cube first)"
  exit 2
fi

echo "=== cube reached-pose DP  JOB=$SLURM_JOB_ID ==="
echo "    labels = $LABELS"
echo "    steps  = $GRADIENT_STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash "$REPO/scripts/sail/train_reached_dp.sh"
echo "=== DONE rc=$? ==="
ls -lh "$STABLEWM_HOME/cube/sail/dp_reached/models/" 2>/dev/null | tail -3
