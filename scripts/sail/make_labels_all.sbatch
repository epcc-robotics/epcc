#!/usr/bin/env bash
# SAIL step 1/3 — reached-pose + AWE precision labels for all three tasks.
#
# cube  -> scripts/sail/make_labels.py        (paths made env-driven)
# plain
# slippery -> scripts/sail/make_labels_pusht.py  (mirror; 2-D agent, pos_only AWE)
#
# CPU only: DBSCAN + the AWE DP are numpy/sklearn, no GPU. Frames are never
# read — only pos_agent / proprio_effector_* — so this is I/O-light despite the
# 12 GB sources.
#
# SMOKE=1 runs 5 episodes per task into *_smoke.h5 and prints waypoint counts,
# which is the number to sanity-check before committing: too few waypoints means
# err_threshold is too loose and the reached targets degenerate to a straight
# line; too many means every frame is a keyframe and precision saturates at 1.
#
#SBATCH --job-name=sail_labels
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
unset LD_PRELOAD
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPO=$REPO_ROOT
PY=python
export AWE_ROOT=${AWE_ROOT:-$REPO_ROOT/third_party/awe}
H=$STABLEWM_HOME

NEPS=(); SFX=""
if [ "${SMOKE:-0}" = "1" ]; then NEPS=(--n-eps 5); SFX="_smoke"; echo ">>> SMOKE: 5 episodes/task"; fi
W=${WORKERS:-16}

cd "$REPO"
echo "=== SAIL labels   JOB=$SLURM_JOB_ID   AWE_ROOT=$AWE_ROOT ==="
fail=0

TASKS=${TASKS:-"plain slippery cube"}
echo ">>> tasks: $TASKS"
for task in plain slippery; do
  case " $TASKS " in *" $task "*) ;; *) echo "skip $task"; continue;; esac
  echo ""
  echo "########## $task ##########"
  # MUST be an array, not "": `"${out[@]}"` on an empty STRING expands to one
  # empty argument, which argparse rejects as "unrecognized arguments".
  out=()
  [ -n "$SFX" ] && out=(--out "$H/$([ $task = slippery ] && echo slippery_pusht || echo pusht)/sail/awe/${task}_reached_precision_park${SFX}.h5")
  $PY scripts/sail/make_labels_pusht.py --task "$task" --workers "$W" \
      "${NEPS[@]}" "${out[@]}" || { echo "FAILED $task"; fail=1; }
done

echo ""
case " $TASKS " in *" cube "*) ;; *) echo "skip cube"; echo "=== DONE fail=$fail ==="; exit $fail;; esac
echo "########## cube ##########"
SAIL_OUT="$H/cube/sail/awe/cube_reached_precision_c03${SFX}.h5" \
  $PY scripts/sail/make_labels.py --workers "$W" \
      ${SMOKE:+--n-eps 5} \
      --out "$H/cube/sail/awe/cube_reached_precision_c03${SFX}.h5" \
  || { echo "FAILED cube"; fail=1; }

echo ""
echo "=== produced ==="
ls -la "$H"/*/sail/awe/ 2>/dev/null
echo "=== DONE fail=$fail ==="
exit $fail
