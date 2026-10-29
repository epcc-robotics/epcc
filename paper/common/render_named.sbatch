#!/usr/bin/env bash
#SBATCH --job-name=render_named
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPRO=$REPO_ROOT
cd "$REPRO"
S=findings/260907_final_dtg_s1_warm/scripts
echo "### Fig 3a friction"
FIG_W=3.54 FIG_H=1.80 FIG_PT=8 TITLES=1 SUF=_named python -u $S/plot_friction_plain3way_1to1.py 2>&1 | tail -3
echo "### Fig 3b phase"
FIG_W=3.54 FIG_H=1.80 FIG_PT=8 PHASE_XLAB="motion phase" SUF=_named python -u $S/contact_split_3task_1to1.py 2>&1 | tail -3
echo "### Fig 4 cgrid"
FIG_W=3.54 FIG_H=2.60 FIG_PT=8 COMPACT=1 SHOW_PREV=0 RIGHT_RECT=0.955 SUF=_named python -u $S/plot_cgrid_sweep_1to1.py 2>&1 | tail -3
echo "=== DONE render rc=$? ==="
