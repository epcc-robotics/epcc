#!/usr/bin/env bash
# Fig. 3a (friction sweep): run the paper's evaluation cells (config/eval/cells/paper/fig3a/*.yaml).
#   paper/fig3/run_cells.sh [--slurm] [cell ...] [-- hydra overrides]
# Cells default to all of them; with --slurm each cell is submitted as one job.
# Outputs: results/fig3a/<cell>/ (results.json, rollout_traj.npz, ...). Score with
#   python paper/common/score_runs.py --out results/fig3a/scores.json results/fig3a/<cell> ...
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
source "$REPO_ROOT/scripts/env.sh" || exit 1
MODE=""; if [ "${1:-}" = "--slurm" ]; then MODE=--slurm; shift; fi
ALL=(epcc_d000 epcc_d010 epcc_d020 epcc_d030 epcc_d040 epcc_d050 epcc_d060 epcc_d070 epcc_d080 sail_d000 sail_d010 sail_d020 sail_d030 sail_d040 sail_d050 sail_d060 sail_d070 sail_d080 sup_d000 sup_d010 sup_d020 sup_d030 sup_d040 sup_d050 sup_d060 sup_d070 sup_d080)
CELLS=(); OVR=()
while [ $# -gt 0 ]; do if [ "$1" = "--" ]; then shift; OVR=("$@"); break; fi; CELLS+=("$1"); shift; done
if [ ${#CELLS[@]} -eq 0 ]; then CELLS=("${ALL[@]}"); fi
for c in "${CELLS[@]}"; do
  case "$c" in
    epcc_d000|epcc_d010|epcc_d020|epcc_d030|epcc_d040|epcc_d050|epcc_d060|epcc_d070|epcc_d080) t=04:00:00 ;;   # planner cells
    *) t=01:30:00 ;;                                   # baseline cells
  esac
  SLURM_TIME=$t LOGS_SUBDIR=fig3a "$REPO_ROOT/paper/run_cell.sh" $MODE "paper/fig3a/$c" "$c" "${OVR[@]}" || exit 1
done
