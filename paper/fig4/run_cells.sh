#!/usr/bin/env bash
# Fig. 4 (expert-play complementarity): run the paper's evaluation cells (config/eval/cells/paper/fig4/*.yaml).
#   paper/fig4/run_cells.sh [--slurm] [cell ...] [-- hydra overrides]
# Cells default to all of them; with --slurm each cell is submitted as one job.
# Outputs: results/fig4/<cell>/ (results.json, rollout_traj.npz, ...). Score with
#   python paper/common/score_runs.py --out results/fig4/scores.json results/fig4/<cell> ...
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
source "$REPO_ROOT/scripts/env.sh" || exit 1
MODE=""; if [ "${1:-}" = "--slurm" ]; then MODE=--slurm; shift; fi
ALL=(cube_CTRL cube_D03 cube_D075 cube_D12 cube_DEMO pusht_CTRL pusht_D05 pusht_D15 pusht_D25 pusht_DEMO slippery_pusht_CTRL slippery_pusht_D05 slippery_pusht_D15 slippery_pusht_D25 slippery_pusht_DEMO)
CELLS=(); OVR=()
while [ $# -gt 0 ]; do if [ "$1" = "--" ]; then shift; OVR=("$@"); break; fi; CELLS+=("$1"); shift; done
if [ ${#CELLS[@]} -eq 0 ]; then CELLS=("${ALL[@]}"); fi
for c in "${CELLS[@]}"; do
  case "$c" in
    cube_CTRL|cube_D03|cube_D075|cube_D12|cube_DEMO|pusht_CTRL|pusht_D05|pusht_D15|pusht_D25|pusht_DEMO|slippery_pusht_CTRL|slippery_pusht_D05|slippery_pusht_D15|slippery_pusht_D25|slippery_pusht_DEMO) t=04:00:00 ;;   # planner cells
    *) t=01:30:00 ;;                                   # baseline cells
  esac
  SLURM_TIME=$t LOGS_SUBDIR=fig4 "$REPO_ROOT/paper/run_cell.sh" $MODE "paper/fig4/$c" "$c" "${OVR[@]}" || exit 1
done
