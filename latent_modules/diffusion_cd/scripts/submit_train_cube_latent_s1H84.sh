#!/usr/bin/env bash
# Train the paper's Cube DTG (contour generator): stride 1, horizon 84, lambda_vel 3.
#   CFG=A_mm_s1H84_lv30 sbatch latent_modules/diffusion_cd/scripts/submit_train_cube_latent_s1H84.sh
# Run from the repo root. Configs: latent_modules/diffusion_cd/configs/cube_latent/; outputs under
# $STABLEWM_HOME/train/dtg/. The eval cells load the step selected by validation (docs/DATA.md).
#SBATCH --job-name=dtg_cube_s1H84
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
set -e
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
CFG=${CFG:?set CFG, e.g. CFG=A_mm_s1H84_lv30}
ENV_SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")  # site-packages of the active env (pip nvidia libs)
NVIDIA_LIB_PATHS=""
for d in "$ENV_SITE"/nvidia/*/lib; do [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"; done
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:-}"
cd "$REPO_ROOT/latent_modules/diffusion_cd"
echo "[dtg] Cube DTG (stride 1, H84) | config=$CFG | host=$(hostname)"
python pipelines/train_latent_velloss.py --config-path ../configs/cube_latent --config-name "$CFG" "$@"
echo "[dtg] $CFG done at $(date)"
