#!/usr/bin/env bash
# Appendix data-scaling sweep: run the paper's evaluation cells (config/eval/cells/paper/scaling/*.yaml).
#   paper/scaling/run_cells.sh [--slurm] [cell ...] [-- hydra overrides]
# Cells default to all of them; with --slurm each cell is submitted as one job.
# Outputs: results/scaling/<cell>/ (results.json, rollout_traj.npz, ...). Score with
#   python paper/common/score_runs.py --out results/scaling/scores.json results/scaling/<cell> ...
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
source "$REPO_ROOT/scripts/env.sh" || exit 1
MODE=""; if [ "${1:-}" = "--slurm" ]; then MODE=--slurm; shift; fi
ALL=(cube_f1 cube_f128 cube_f16 cube_f256 cube_f256_smoke cube_f32 cube_f4 cube_f64 pusht_f1 pusht_f128 pusht_f16 pusht_f256 pusht_f256_smoke pusht_f32 pusht_f4 pusht_f64 slippery_pusht_f1 slippery_pusht_f128 slippery_pusht_f16 slippery_pusht_f256 slippery_pusht_f256_smoke slippery_pusht_f32 slippery_pusht_f4 slippery_pusht_f64)
CELLS=(); OVR=()
while [ $# -gt 0 ]; do if [ "$1" = "--" ]; then shift; OVR=("$@"); break; fi; CELLS+=("$1"); shift; done
if [ ${#CELLS[@]} -eq 0 ]; then CELLS=("${ALL[@]}"); fi
for c in "${CELLS[@]}"; do
  case "$c" in
    cube_f1|cube_f128|cube_f16|cube_f256|cube_f256_smoke|cube_f32|cube_f4|cube_f64|pusht_f1|pusht_f128|pusht_f16|pusht_f256|pusht_f256_smoke|pusht_f32|pusht_f4|pusht_f64|slippery_pusht_f1|slippery_pusht_f128|slippery_pusht_f16|slippery_pusht_f256|slippery_pusht_f256_smoke|slippery_pusht_f32|slippery_pusht_f4|slippery_pusht_f64) t=04:00:00 ;;   # planner cells
    *) t=01:30:00 ;;                                   # baseline cells
  esac
  SLURM_TIME=$t LOGS_SUBDIR=scaling "$REPO_ROOT/paper/run_cell.sh" $MODE "paper/scaling/$c" "$c" "${OVR[@]}" || exit 1
done
