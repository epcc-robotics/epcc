#!/usr/bin/env bash
# SuP scheduler rebuild on the PARK demo generation  (task #3, 2026-08-01)
#
# Why. Every other row of our table — DP, DTG-MPCC, SkiP — is trained on the
# park (post-success stabilise + back-off tail) demo sets. SuP's ADM/IQL were
# built from the non-park ones, so it was sitting on a different generation.
#
# Why it is not cosmetic. train_iql.py:24 picks the safety threshold as
#   eps = quantile(E[K>1], eps_q=0.4)
# a quantile over the WHOLE D' deviation distribution. The park tail is free
# retreat motion with almost no deviation, so adding it drags the quantile down
# and re-labels contact-phase states as unsafe. The effect is global, not
# tail-local — which is exactly why both variants are built here rather than
# just overwriting.
#
# The ADM is NOT retrained: it learns from the WM corpus, not the demos.
#
# Outputs (10 = 3 tasks x 2 channels park, + 2 pushT-family tasks x 2 channels
# non-park control). cube has no park/non-park split — its demo set has been
# postpad since collection, which is where park was ported from.
#   $STABLEWM_HOME/<task>/sup/iql_{robot,full}_{park,nonpark}_dataset.npz
#   $STABLEWM_HOME/<task>/sup/iql_{robot,full}_{park,nonpark}.pt
#
# Chunk length stays at the default C=15: every gated cell runs
# action_block=1 x receding_horizon=15, and the new slippery CleanDiffuser
# backbone (horizon 16) will use the same, so README known-issue #1 — the
# 39-vs-37 truncation on the old robomimic slippery path — disappears with the
# backbone swap instead of needing --chunk 16.
#
#SBATCH --job-name=sup_iql_park
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
unset LD_PRELOAD
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPO=$REPO_ROOT
PY=python
H=$STABLEWM_HOME

# demo h5 per (task, generation). "park" is the generation the rest of the
# table uses; "nonpark" reproduces what the shipped iql_*.pt were built from.
declare -A DEMO=(
  [cube:park]="$H/cube/data/cube_multiview_diffusion_c03.h5"
  [pusht:park]="$H/datasets/expert_slow_025_n200_centered_v2_park.h5"
  [pusht:nonpark]="$H/pusht/data/expert_slow_025_n200.h5"
  [slippery:park]="$H/slippery_pusht/data/expert_slow_050_n200_d035_park.h5"
  [slippery:nonpark]="$H/slippery_pusht/data/expert_slow_050_n200_d035.h5"
)
declare -A SUPDIR=([cube]="$H/cube/sup" [pusht]="$H/pusht/sup" [slippery]="$H/slippery_pusht/sup")

# SMOKE=1 — one task per family, 5 episodes, products written under _smoke tags
# so nothing real is overwritten. Two minutes instead of finding a path typo
# three hours into the full sweep.
KEYS=("${!DEMO[@]}"); EXTRA=(); TAGSFX=""
# ONLY=<task>:<gen> runs a single (task, generation) pair — both channels. The
# full sweep is ~10 x 40 min of D' synthesis, which does not fit one walltime
# comfortably; one job per pair also means a single failure costs one pair.
if [ -n "${ONLY:-}" ]; then
  KEYS=("$ONLY")
  echo ">>> ONLY: $ONLY"
fi
if [ "${SMOKE:-0}" = "1" ]; then
  KEYS=(pusht:park slippery:park cube:park)
  EXTRA=(--max_ep 5); TAGSFX="_smoke"
  echo ">>> SMOKE: ${KEYS[*]}  max_ep=5  tag suffix=$TAGSFX"
fi

cd "$REPO"
SUMMARY="$REPO_ROOT/logs/sup_iql_park_${SLURM_JOB_ID}_summary.txt"
: > "$SUMMARY"
echo "=== SuP D' + IQL rebuild   JOB=$SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fail=0
for key in "${KEYS[@]}"; do
  task="${key%%:*}"; gen="${key##*:}${TAGSFX}"; demo="${DEMO[$key]}"; d="${SUPDIR[$task]}"
  if [ ! -e "$demo" ]; then echo "SKIP $key — demo missing: $demo"; fail=1; continue; fi
  for ch in robot full; do
    echo ""
    echo "########## $task / $ch / $gen ##########"
    $PY scripts/sup/build_iql_dataset.py --task "$task" --channel "$ch" \
        --demo "$demo" --tag "_${gen}" "${EXTRA[@]}" || { echo "BUILD FAILED $key/$ch"; fail=1; continue; }
    # train_iql prints "[IQL] <name> N=.. sd=.. eps=X safe=Y%" and a pick%
    # histogram. Both are the answer to "does park move the scheduler", so tee
    # them into a summary instead of leaving them buried in a 10-run log.
    $PY scripts/sup/train_iql.py \
        --dataset "$d/iql_${ch}_${gen}_dataset.npz" \
        --out     "$d/iql_${ch}_${gen}.pt" --eps_q 0.4 \
      | tee /dev/stderr | sed "s|^|$task/$ch/$gen  |" >> "$SUMMARY" \
      || { echo "IQL FAILED $key/$ch"; fail=1; }
  done
done

echo ""
echo "=== eps + chosen-k distribution, park vs non-park ==="
echo "    (eps down => threshold tighter => scheduler more conservative => smaller k)"
sort "$SUMMARY"
echo "=== DONE fail=$fail ==="
exit $fail
