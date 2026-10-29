#!/usr/bin/env bash
# SAIL step 2/3 for the PushT family — train the reached-pose DP.
#
# scripts/sail/train_reached_dp.sh is cube-only: it calls
# submit_train_dp_cube_aw.sh and hardcodes action_dim=6. This is the PushT
# equivalent (action_dim=3), left as a separate launcher so that script stays
# the cube path.
#
#   TASK=slippery|plain  sbatch train_dp_reached_pusht.sbatch
#
# Requires the labels from scripts/sail/make_labels_pusht.py to exist.
#
#SBATCH --job-name=dp_reached
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
unset LD_PRELOAD
PY=python
LD_ROOT=$REPO_ROOT/latent_modules/diffusion_cd
ENV_SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")  # site-packages of the active env (pip nvidia libs)
NV=""; for d in "$ENV_SITE"/nvidia/*/lib; do [ -d "$d" ] && NV="${d}:${NV}"; done
export LD_LIBRARY_PATH="${NV}${LD_LIBRARY_PATH:-}"

TASK=${TASK:-slippery}
case "$TASK" in
  slippery) LBL="$STABLEWM_HOME/slippery_pusht/sail/awe/slippery_reached_precision_park.h5" ;;
  # make_labels_pusht.py writes the PLAIN file as pusht_*, not plain_* — the
  # task key is "plain" but the asset directory is pusht/, and the filename
  # follows the directory.
  plain)    LBL="$STABLEWM_HOME/pusht/sail/awe/pusht_reached_precision_park.h5" ;;
  *) echo "TASK must be slippery|plain (got $TASK)"; exit 2 ;;
esac
LBL=${LABELS:-$LBL}   # LABELS= lets the smoke run point at the 5-episode file
[ -f "$LBL" ] || { echo "labels missing: $LBL  (run scripts/sail/make_labels_pusht.py --task $TASK)"; exit 2; }

STEPS=${STEPS:-100000}
if [ "${SMOKE:-0}" = "1" ]; then STEPS=200; fi

cd "$LD_ROOT"
echo "=== SAIL reached-pose DP  task=$TASK  JOB=$SLURM_JOB_ID ==="
echo "    labels = $LBL"
echo "    steps  = $STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY pipelines/dp_image.py \
    --config-path ../configs/dp/pusht_sv/chi_unet \
    --config-name pusht_sv_image_reached \
    ++dataset_path="$LBL" \
    ++exp_name="reached_${TASK}_park${SMOKE:+_smoke}" \
    ++gradient_steps="$STEPS" \
    "$@"
echo "=== DONE rc=$? ==="
