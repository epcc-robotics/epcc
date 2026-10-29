#!/usr/bin/env bash
# Common prologue for every launcher in this repository. Usage (first lines of a script):
#   REPO_ROOT=${REPO_ROOT:-...}; source "$REPO_ROOT/scripts/env.sh" || exit 1
# Scripts run directly derive REPO_ROOT from their own path; *.sbatch files are executed from
# a spool copy by Slurm, so they use the submit directory (submit them from the repo root,
# ideally through scripts/submit.sh).
if [ -z "${REPO_ROOT:-}" ] || [ ! -f "$REPO_ROOT/scripts/env.sh" ]; then
  echo "env.sh: REPO_ROOT='${REPO_ROOT:-}' is not the repository root. Run from the repo root or set REPO_ROOT." >&2
  return 1 2>/dev/null || exit 1
fi
REPO_ROOT=$(cd "$REPO_ROOT" && pwd) || return 1
export REPO_ROOT
if [ -f "$REPO_ROOT/cluster.env" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/cluster.env" || { echo "env.sh: failed to source $REPO_ROOT/cluster.env" >&2; return 1; }
fi
if [ -n "${VENV:-}" ]; then
  [ -f "$VENV/bin/activate" ] || { echo "env.sh: VENV=$VENV has no bin/activate" >&2; return 1; }
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" || { echo "env.sh: failed to activate $VENV" >&2; return 1; }
fi
if [ -n "${CUDA_MODULE:-}" ]; then
  command -v module >/dev/null 2>&1 || { echo "env.sh: CUDA_MODULE=$CUDA_MODULE is set but there is no 'module' command" >&2; return 1; }
  module load "$CUDA_MODULE" || { echo "env.sh: 'module load $CUDA_MODULE' failed" >&2; return 1; }
fi
export MUJOCO_GL=${MUJOCO_GL:-egl}
export WANDB_MODE=${WANDB_MODE:-disabled}
export PYTHONUNBUFFERED=1
# Repository packages that are imported by name but not pip-installed:
# latent_diffusion (latent_modules/diffusion) and latent_diffusion_cd (DTG trainer and loader).
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/latent_modules/diffusion:$REPO_ROOT/latent_modules/diffusion_cd${PYTHONPATH:+:$PYTHONPATH}"
if [ -z "${STABLEWM_HOME:-}" ]; then
  echo "env.sh: set STABLEWM_HOME (data root, see docs/DATA.md) in cluster.env or the environment" >&2
  return 1
fi
export STABLEWM_HOME
mkdir -p "$REPO_ROOT/logs" || return 1
