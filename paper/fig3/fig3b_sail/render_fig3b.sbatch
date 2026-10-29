#!/usr/bin/env bash
# Fast half of Fig. 3b (seconds): sbatch render_fig3b.sbatch <full|robot> <paired4|own> <0|1 CI>
# paper sizing; output ../render/tableI[_suprobot][_own]/
#SBATCH --job-name=fig3b_render
#SBATCH --nodes=1 --cpus-per-task=1 --mem=4G --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPRO=$REPO_ROOT
cd "$REPRO" || exit 1
S=findings/261022_fig3b_sail/scripts
PAPER="FIG_W=3.54 FIG_H=1.80 FIG_PT=8 RIGHT_RECT=0.94 CUBE_MAP=precision SUF=_prec"
env $PAPER SUP_MODE=${1:-full} AGG=${2:-paired4} ERR=${3:-0} PHASE_XLAB='motion phase' python -u $S/render_fig3b.py
