#!/usr/bin/env bash
# Submit any *.sbatch launcher of this repository with the site settings applied:
#   scripts/submit.sh [sbatch options] path/to/launcher.sbatch [launcher args]
# It loads cluster.env, routes the job by the launcher's own #SBATCH --gres line (GPU jobs:
# GPU_PARTITION / GPU_GRES, CPU jobs: CPU_PARTITION; SBATCH_ACCOUNT is read by sbatch),
# creates logs/ (the launchers write logs/%x_%j.out) and submits from the repo root.
# Options given on the command line override these defaults.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -f "$REPO_ROOT/cluster.env" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/cluster.env"
fi
mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT"
script=""
for a in "$@"; do case "$a" in *.sbatch|*.sh) script=$a ;; esac; done
site=()
if [ -n "$script" ] && grep -q -E '^#SBATCH[[:space:]]+--gres=gpu' "$script"; then
  [ -n "${GPU_PARTITION:-}" ] && site+=(--partition="$GPU_PARTITION")
  [ -n "${GPU_GRES:-}" ] && site+=(--gres="$GPU_GRES")
else
  [ -n "${CPU_PARTITION:-}" ] && site+=(--partition="$CPU_PARTITION")
fi
exec sbatch "${site[@]}" --export=ALL,REPO_ROOT="$REPO_ROOT" "$@"
