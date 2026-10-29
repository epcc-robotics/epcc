#!/usr/bin/env bash
# PACE launcher for the ported cube play collector (scripts/collect_multiview_WM_cube.py
# in le_wm_v2) with the afterclip pace knob (SpeedScaledExpertPolicy).
# Env setup mirrors le-wm/scripts/submit_collect_multiview_WM_cube.sh (the job
# that produced the champion corpus) — le-wm/env python + swm_srl_v2 stack.
#
# Usage (comma-free --export values only):
#   sbatch --export=ALL,MODE=validate,C=1.2 scripts/data/submit_collect_cube_play_pace.sbatch
#   sbatch --export=ALL,MODE=collect,C=0.75,NUM_TRAJ=1000,SEED=3073 ...
# c<1 auto-lengthens episodes (round(200/c)) to keep goal-resample density.
#SBATCH --job-name=collect_cube_play
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1

cd "$REPO_ROOT"

MODE=${MODE:-validate}       # validate: n_val eps + side-by-side videos / collect: full
C=${C:-1.0}
NUM_TRAJ=${NUM_TRAJ:-1300}   # ~256k frames at c>=1 (201 fr/ep); c=0.75 -> ~960, c=0.3 -> ~385
SEED=${SEED:-3073}           # champion corpus used 3072 — never reuse
NUM_ENVS=${NUM_ENVS:-10}

echo "== collect cube play  JOB=$SLURM_JOB_ID  mode=$MODE c=$C n=$NUM_TRAJ seed=$SEED =="
python -c "import stable_worldmodel as s; print('SWM ->', s.__file__)" || exit 1

python scripts/collect_multiview_WM_cube.py \
    mode="$MODE" \
    slowdown_factor="$C" \
    num_traj="$NUM_TRAJ" \
    seed="$SEED" \
    world.num_envs="$NUM_ENVS" \
    hydra.run.dir=logs_dist_shift_collect/hydra/collect_cube_play_${SLURM_JOB_ID}
RC=$?
echo "== collect rc=$RC =="
exit $RC
