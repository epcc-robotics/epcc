#!/usr/bin/env bash
# Table I: run the paper's evaluation cells (config/eval/cells/paper/table1/*.yaml).
#   paper/table1/run_cells.sh [--slurm] [cell ...] [-- hydra overrides]
# Cells default to all of them; with --slurm each cell is submitted as one job.
# Outputs: results/table1/<cell>/ (results.json, rollout_traj.npz, ...). Score with
#   python paper/common/score_runs.py --out results/table1/scores.json results/table1/<cell> ...
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
source "$REPO_ROOT/scripts/env.sh" || exit 1
MODE=""; if [ "${1:-}" = "--slurm" ]; then MODE=--slurm; shift; fi
ALL=(cube_dp cube_epcc cube_playgoal cube_sail_k1.5 cube_sail_k2.0 cube_sail_k2.5 cube_sail_k3.0 cube_skip cube_sup pusht_dp pusht_epcc pusht_playgoal pusht_sail_k1.5 pusht_sail_k2.0 pusht_sail_k2.5 pusht_sail_k3.0 pusht_skip pusht_sup slippery_pusht_dp slippery_pusht_epcc slippery_pusht_playgoal slippery_pusht_sail_k1.5 slippery_pusht_sail_k2.0 slippery_pusht_sail_k2.5 slippery_pusht_sail_k3.0 slippery_pusht_skip slippery_pusht_sup)
CELLS=(); OVR=()
while [ $# -gt 0 ]; do if [ "$1" = "--" ]; then shift; OVR=("$@"); break; fi; CELLS+=("$1"); shift; done
if [ ${#CELLS[@]} -eq 0 ]; then CELLS=("${ALL[@]}"); fi
for c in "${CELLS[@]}"; do
  case "$c" in
    cube_epcc|pusht_epcc|slippery_pusht_epcc) t=04:00:00 ;;   # planner cells
    *) t=01:30:00 ;;                                   # baseline cells
  esac
  SLURM_TIME=$t LOGS_SUBDIR=table1 "$REPO_ROOT/paper/run_cell.sh" $MODE "paper/table1/$c" "$c" "${OVR[@]}" || exit 1
done
