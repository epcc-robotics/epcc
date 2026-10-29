#!/usr/bin/env bash
# Run one evaluation cell: a Hydra config under config/eval/cells, e.g. a paper cell
# paper/table1/pusht_epcc (config/eval/cells/paper/table1/pusht_epcc.yaml).
#
#   paper/run_cell.sh [--slurm] <cell> [run_name] [hydra overrides ...]
#
#   paper/run_cell.sh paper/table1/pusht_epcc                       one GPU, here
#   paper/run_cell.sh paper/table1/pusht_epcc smoke task.num_eval=2 two episodes
#   paper/run_cell.sh --slurm paper/table1/pusht_epcc               submit as a Slurm job
#
# Output: $RESULTS_DIR/$LOGS_SUBDIR/<run_name>/ (results.json, rollout_traj.npz, ...);
# defaults results/runs/<cell name>_n50. An existing run directory with results.json is
# refused unless OVERWRITE=1. Score it with paper/common/score_runs.py.
# Environment knobs: RESULTS_DIR, LOGS_SUBDIR, EVAL_CONFIG_DIR (relative to scripts/baseline/),
# and for --slurm SLURM_TIME / SLURM_MEM / SLURM_CPUS and SLURM_ARGS (extra sbatch options, e.g.
# "--exclude=<node>"); account, GPU partition and gres come from cluster.env.
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$REPO_ROOT/scripts/env.sh" || exit 1

SLURM=0
if [ "${1:-}" = "--slurm" ]; then SLURM=1; shift; fi
CELL=${1:?usage: paper/run_cell.sh [--slurm] <cell> [run_name] [hydra overrides ...]}; shift
RUN_NAME="${CELL##*/}_n50"
if [ $# -gt 0 ] && [[ "$1" != *=* ]]; then RUN_NAME=$1; shift; fi
case "$RUN_NAME" in */*|"") echo "run_cell.sh: run_name must be non-empty and contain no '/': $RUN_NAME" >&2; exit 2 ;; esac
RESULTS_DIR=${RESULTS_DIR:-results}
case "$RESULTS_DIR" in /*) ;; *) RESULTS_DIR="$REPO_ROOT/$RESULTS_DIR" ;; esac
LOGS_SUBDIR=${LOGS_SUBDIR:-runs}
export EVAL_CONFIG_DIR=${EVAL_CONFIG_DIR:-../../config/eval/cells}
OUT_DIR="$RESULTS_DIR/$LOGS_SUBDIR/$RUN_NAME"
if [ -f "$OUT_DIR/results.json" ] && [ "${OVERWRITE:-0}" != 1 ]; then
  echo "run_cell.sh: $OUT_DIR already has results.json (set OVERWRITE=1 to replace it)" >&2; exit 3
fi

if [ "$SLURM" = 1 ]; then
  export RESULTS_DIR LOGS_SUBDIR
  # sbatch copies this script to a spool directory and passes the arguments through unchanged.
  sbatch --parsable --job-name="$RUN_NAME" --nodes=1 \
    --cpus-per-task="${SLURM_CPUS:-4}" --mem="${SLURM_MEM:-32G}" --time="${SLURM_TIME:-02:30:00}" \
    ${GPU_PARTITION:+--partition="$GPU_PARTITION"} --gres="${GPU_GRES:-gpu:1}" \
    --output="$REPO_ROOT/logs/%x_%j.out" \
    --export=ALL,REPO_ROOT="$REPO_ROOT" ${SLURM_ARGS:-} \
    "$REPO_ROOT/paper/run_cell.sh" "$CELL" "$RUN_NAME" "$@"
  exit $?
fi

cd "$REPO_ROOT" || exit 1
echo "=== cell=$CELL run_name=$RUN_NAME out=$OUT_DIR STABLEWM_HOME=$STABLEWM_HOME job=${SLURM_JOB_ID:-local}"
python -c "import stable_worldmodel as s, cleandiffuser as c; print('stable_worldmodel ->', s.__file__); print('cleandiffuser ->', c.__file__)" || exit 1
python scripts/baseline/eval_unified.py \
  --config-name "$CELL" \
  ++output.findings_dir="$RESULTS_DIR" \
  ++output.logs_subdir="$LOGS_SUBDIR" \
  ++output.run_name="$RUN_NAME" \
  hydra.run.dir="$REPO_ROOT/logs/hydra/${RUN_NAME}_${SLURM_JOB_ID:-local}" \
  "$@"
rc=$?
echo "=== done cell=$CELL rc=$rc"
exit $rc
