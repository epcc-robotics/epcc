#!/usr/bin/env bash
# Collect the Cube slow-expert demonstrations (Hydra config scripts/config/multiview_diffusion_cube.yaml).
#   scripts/submit.sh scripts/submit_collect_multiview_diffusion_cube.sh [validate|collect] [hydra overrides]
#SBATCH --job-name=mv_diff_cube
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
cd "$REPO_ROOT" || exit 1

# --- MODE ---
# validate: 5 episodes + side-by-side videos
# collect:  full 2000 episodes (slow expert c=0.3, fixed goal)
MODE=${1:-collect}
SLOWDOWN=${SLOWDOWN:-1.0}
NUM_TRAJ=${NUM_TRAJ:-100}
SUFFIX="${SUFFIX:-}"
S=""
[ -n "$SUFFIX" ] && S="_${SUFFIX}"
NAME=${NAME:-cube_multiview_diffusion_c${SLOWDOWN}${S}}

# DEPRECATED: TARGET_EP_LEN no longer used. Variable-length episodes terminate
# POST_SUCCESS_FRAMES (=100) steps after first success_reached, with scripted
# natural retreat motion (no zero-padding).
echo "[submit_collect] MODE=$MODE SLOWDOWN=$SLOWDOWN NUM_TRAJ=$NUM_TRAJ NAME=$NAME (variable-length)"

python scripts/collect_multiview_diffusion_cube.py \
    mode=$MODE \
    slowdown_factor=$SLOWDOWN \
    num_traj=$NUM_TRAJ \
    +name="$NAME" \
    "${@:2}"
