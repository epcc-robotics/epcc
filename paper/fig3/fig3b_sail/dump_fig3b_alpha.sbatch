#!/usr/bin/env bash
# Slow half of Fig. 3b: per-episode alpha -> ../data/alpha_per_episode_tableI.json (run once).
#SBATCH --job-name=fig3b_dump
#SBATCH --nodes=1 --cpus-per-task=4 --mem=32G --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPRO=$REPO_ROOT
cd "$REPRO" || exit 1
S=findings/261022_fig3b_sail/scripts
PAPER="FIG_W=3.54 FIG_H=1.80 FIG_PT=8 RIGHT_RECT=0.94 CUBE_MAP=precision SUF=_prec"
C=$REPRO/findings/260907_final_dtg_s1_warm
export EPCC_PUSHT=$C/runs/table1/t1_plain_M2_s1warm EPCC_CUBE=$C/runs/table1/t1_cube_M2_s1warm EPCC_SLIP=$C/runs/table1/t1_slip_M1_s1warm
python -u $S/dump_fig3b_alpha.py
