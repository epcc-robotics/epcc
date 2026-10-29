#!/usr/bin/env bash
# Table II: run the paper's evaluation cells (config/eval/cells/paper/table2/*.yaml).
#   paper/table2/run_cells.sh [--slurm] [cell ...] [-- hydra overrides]
# Cells default to all of them; with --slurm each cell is submitted as one job.
# Outputs: results/table2/<cell>/ (results.json, rollout_traj.npz, ...). Score with
#   python paper/common/score_runs.py --out results/table2/scores.json results/table2/<cell> ...
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
source "$REPO_ROOT/scripts/env.sh" || exit 1
MODE=""; if [ "${1:-}" = "--slurm" ]; then MODE=--slurm; shift; fi
ALL=(cube_full cube_fullcand cube_refine cube_refinesup cube_select pusht_full pusht_fullcand pusht_refine pusht_refinesup pusht_select slippery_pusht_full slippery_pusht_fullcand slippery_pusht_refine slippery_pusht_refinesup slippery_pusht_select)
CELLS=(); OVR=()
while [ $# -gt 0 ]; do if [ "$1" = "--" ]; then shift; OVR=("$@"); break; fi; CELLS+=("$1"); shift; done
if [ ${#CELLS[@]} -eq 0 ]; then CELLS=("${ALL[@]}"); fi
for c in "${CELLS[@]}"; do
  case "$c" in
    cube_full|cube_fullcand|cube_refine|cube_refinesup|cube_select|pusht_full|pusht_fullcand|pusht_refine|pusht_refinesup|pusht_select|slippery_pusht_full|slippery_pusht_fullcand|slippery_pusht_refine|slippery_pusht_refinesup|slippery_pusht_select) t=04:00:00 ;;   # planner cells
    *) t=01:30:00 ;;                                   # baseline cells
  esac
  SLURM_TIME=$t LOGS_SUBDIR=table2 "$REPO_ROOT/paper/run_cell.sh" $MODE "paper/table2/$c" "$c" "${OVR[@]}" || exit 1
done
