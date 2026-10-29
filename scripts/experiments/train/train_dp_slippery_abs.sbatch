#!/usr/bin/env bash
# Slippery PushT — CleanDiffuser DP with ABSOLUTE agent-pose actions.
#
# Replaces the Robomimic DP backbone with CleanDiffuser (2026-07-31), keeping Slippery's absolute
# action abstraction so DP / SuP / SkiP / SAIL there share one abstraction.
# The backbone swap moves the numbers — they measured up to 30 pp — so the
# Slippery DP baseline row has to be re-run on this checkpoint afterwards.
#
# Dataset checked before launch (job 11596861): _abs_pose matches the h5
# episode-wise pos_agent concat, the horizon target equals pos_agent[a+1] to
# 0.0000, normalized actions land in [-1,1], and no sample crosses an episode
# boundary.
#
#SBATCH --job-name=dp_slip_abs
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

# 🔴 2026-08-01: the first run (job 11596881) trained on the NON-PARK set —
# 81,310 frames instead of the park set's 101,310 — which puts it on a different
# generation from every other row in the table (DP / DTG-MPCC / SkiP are all
# park). Both are wanted in the end: park for the table, non-park as the control
# that pairs with the non-park D'/IQL variant. Hence the two knobs.
#   DATASET=<h5>  EXP_NAME=<name>  sbatch train_dp_slippery_abs.sbatch
DEFAULT_DS=$STABLEWM_HOME/slippery_pusht/data/expert_slow_050_n200_d035_park.h5

cd "$LD_ROOT"
echo "=== slippery CleanDiffuser DP (ABSOLUTE action)  JOB=$SLURM_JOB_ID ==="
echo "    dataset  = ${DATASET:-$DEFAULT_DS}"
echo "    exp_name = ${EXP_NAME:-slip_abs_cd}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY pipelines/dp_image.py \
    --config-path ../configs/dp/pusht_sv/chi_unet \
    --config-name pusht_sv_image_abs \
    ++exp_name="${EXP_NAME:-slip_abs_cd}" \
    ++dataset_path="${DATASET:-$DEFAULT_DS}" \
    "$@"
echo "=== DONE rc=$? ==="
