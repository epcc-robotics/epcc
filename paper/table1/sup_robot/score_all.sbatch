#!/usr/bin/env bash
# Score the robot-only SuP reruns with the paper's own scorers.
#   1) Table I SuP row   (published column must reproduce the paper)
#   2) Fig. 3a           SUP_MODE=published (must reproduce 260907 friction_scores.json) and robot
#   3) Fig. 3b           SAIL_MODE=tableI SUP_MODE=robot, same env as fig3b/README.md
# Nothing is written under 260907_* or 260521_*; Fig. 3b robot output goes to out/tableI_suprobot.
#SBATCH --job-name=sup_robot_score
#SBATCH --nodes=1 --cpus-per-task=4 --mem=32G --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPRO=$REPO_ROOT
cd "$REPRO" || exit 1
S=findings/261023_sup_robot/scripts
C=$REPRO/findings/260907_final_dtg_s1_warm
fail=0

echo "################ 1) Table I SuP"
python -u $S/score_table1_sup.py || fail=1

for m in published robot; do
  echo "################ 2) Fig. 3a  SUP_MODE=$m"
  SUP_MODE=$m FIG_W=3.54 FIG_H=1.80 FIG_PT=8 TITLES=1 SUF=_named \
    python -u $S/plot_friction_plain3way_1to1_sup.py || fail=1
done

echo "################ 3) Fig. 3b  SAIL_MODE=tableI SUP_MODE=robot"
SAIL_MODE=tableI SUP_MODE=robot \
  EPCC_PUSHT=$C/runs/table1/t1_plain_M2_s1warm EPCC_CUBE=$C/runs/table1/t1_cube_M2_s1warm \
  EPCC_SLIP=$C/runs/table1/t1_slip_M1_s1warm \
  FIG_W=3.54 FIG_H=1.80 FIG_PT=8 PHASE_XLAB='motion phase' RIGHT_RECT=0.94 CUBE_MAP=precision SUF=_prec \
  python -u findings/261022_fig3b_sail/scripts/contact_split_3task_sailcheck.py || fail=1

echo "=== DONE fail=$fail ==="
exit $fail
