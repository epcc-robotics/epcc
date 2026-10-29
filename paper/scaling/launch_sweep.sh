#!/usr/bin/env bash
# Submit the scaling-sweep training jobs of one task (RTX 6000) and chain post_train.sbatch
# (select ckpt -> register -> EPCC eval -> score) after each of them.
#
# Budget = max(Table I epochs of the task, 100k steps):
#   slippery : every size 100 x 1000 = 100k  (Table I M1/M2 recipe: 15 epochs, batch 256 ~ 94-104k)
#   cube     : sizes < 1/1: 50 x 2000 = 100k ; 1/1: 93 x 2000 = 186k (= Table I agg_cubeM2, 7 epochs)
# BUILD_DEPS="<denom>:<jobid> ..." makes a training job wait for the CPU job that writes its subset.
#
#   BUILD_DEPS="4:<jobid> 1:<jobid>" paper/scaling/launch_sweep.sh cube [256 64 16 4 1]
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
source "$REPO_ROOT/scripts/env.sh" || exit 1
cd "$REPO_ROOT" || exit 1
F=paper/scaling
TASK="${1:?usage: launch_sweep.sh plain|slippery|cube [denoms...]}"; shift
if [ $# -gt 0 ]; then DENOMS=("$@"); else DENOMS=(256 64 16 4 1); fi
declare -A DEP=()
for kv in ${BUILD_DEPS:-}; do DEP[${kv%%:*}]=${kv##*:}; done

for d in "${DENOMS[@]}"; do
    case "$TASK" in
      slippery) S=$F/train_wm_scaling.sbatch; SPE=1000; EP=100 ;;
      cube)     S=$F/train_wm_scaling_cube.sbatch; SPE=2000
                if [ "$d" = "1" ]; then EP=93; else EP=50; fi ;;
      plain)    S=$F/train_wm_scaling.sbatch
                if [ "$d" = "1" ]; then SPE=2000; EP=126; else SPE=1000; EP=100; fi ;;
      *) echo "unknown task $TASK"; exit 2 ;;
    esac
    name="scal_${TASK}_f${d}"
    depopt=()
    [ -n "${DEP[$d]:-}" ] && depopt=(--dependency=afterok:${DEP[$d]})
    tj=$("$REPO_ROOT/scripts/submit.sh" --parsable "${depopt[@]}" --job-name="$name" \
         --export=ALL,REPO_ROOT="$REPO_ROOT",TASK="$TASK",DENOM="$d",SPE="$SPE",MAX_EPOCHS="$EP" "$S")
    pj=$("$REPO_ROOT/scripts/submit.sh" --parsable --dependency=afterok:"$tj" --job-name="scal_post_${TASK}_f${d}" \
         --export=ALL,REPO_ROOT="$REPO_ROOT",TASK="$TASK",DENOM="$d",TRAIN_JID="$tj" "$F/post_train.sbatch")
    echo "$name -> train $tj ($EP x $SPE = $(( EP * SPE )) steps${DEP[$d]:+, after build ${DEP[$d]}}) ; post $pj"
done
