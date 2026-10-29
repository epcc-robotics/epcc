#!/usr/bin/env bash
# B-set collection: SuP fixed-k rollouts recorded live into play-format h5
# (++eval.collect_h5, see scripts/baseline/collect_rollout_h5.py).
#
# Grid (plan): tasks {pusht, slippery, cube} x k {1,2,3,4,5} x repeats.
# Repeats yield DISTINCT rollouts from the SAME 50 inits (DDPM sampling is
# unseeded — measured 88-96% spread across identical n=50 runs).
# Frames/run ~= plain 24.5k / slippery 29.7k / cube 15.7k.
#
# vs the eval cells, two deliberate collection-mode deltas:
#   - ++world.resolution=224 (pusht tasks): eval cells render at 96 and the
#     wrapper upscales to 224; the WM corpus needs TRUE 224 frames (play
#     parity). DP input is 96-downsampled either way. NOT bit-reproducing the
#     archived fixed-k SR numbers is accepted — this is collection, not eval.
#   - no save_traj (RAM) / no log_scores.
#
# Usage:
#   TASKS="slippery" KS="3" REPS=1 bash scripts/experiments/collect/submit_collect_bset_fixedk.sh  # smoke
#   TASKS="pusht slippery cube" KS="1 2 3 4 5" REPS=4 bash ...                                     # full
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
source "$REPO_ROOT/scripts/env.sh" || exit 1
cd "$REPO_ROOT" || exit 1
H=$STABLEWM_HOME

TASKS=${TASKS:-"slippery"}
KS=${KS:-"3"}
REPS=${REPS:-1}
REP_OFFSET=${REP_OFFSET:-0}          # continue numbering when adding repeats
NUM_EVAL=${NUM_EVAL:-}               # override episodes-per-run (cube: 100 works
                                     # with MEM=96G — n=100 at default 32G host
                                     # RAM gets slurm-OOM-killed during env setup)
MEM=${MEM:-32G}                      # sbatch --mem (host RAM, not GPU)

declare -A CELL=( [pusht]=pusht_sup [slippery]=slippery_pusht_sup_cd_abs [cube]=cube_sup )
declare -A WALLS=( [pusht]="++world.escapable_block=true" [slippery]="++world.escapable_block=true" [cube]="" )
declare -A RES=( [pusht]="++world.resolution=224" [slippery]="++world.resolution=224" [cube]="" )

for task in $TASKS; do
  for k in $KS; do
    for ((r=REP_OFFSET; r<REP_OFFSET+REPS; r++)); do
      RUN="${task}_bset_k${k}_r${r}"
      # seed varies per repeat -> distinct 50-episode INIT SETS per run
      # (r0=42 == the historical fixed-k eval init set, free SR sanity check;
      # DDPM noise is unseeded on top of this either way)
      SEED=$((42 + r))
      NE_OVR=""
      [ -n "$NUM_EVAL" ] && NE_OVR="++task.num_eval=${NUM_EVAL}"
      EXTRA="++policy.sup.score_mode=none ++policy.sup.fixed_k=${k} ++seed=${SEED} ${NE_OVR} ${WALLS[$task]} ${RES[$task]} ++eval.collect_h5=true ++eval.collect_dataset_name=${RUN} ++eval.collect_dir=$H/bset/${task}"
      echo "submit $RUN  cell=${CELL[$task]} mem=$MEM"
      # shellcheck disable=SC2086  # EXTRA is a list of space-free Hydra overrides
      SLURM_MEM=$MEM LOGS_SUBDIR=bset_collect "$REPO_ROOT/paper/run_cell.sh" --slurm "${CELL[$task]}" "$RUN" $EXTRA
    done
  done
done
