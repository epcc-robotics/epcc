#!/usr/bin/env bash
# After a campaign cell finishes: gate check -> archive overrides -> score (cpu-small) -> ledger row.
#   bash finish_cell.sh <category> <RUN_NAME> <jobid> [--cand] [--expect-dtg <path>] [--expect-stats <corpus>] [--no-warm]
# category in {table1, fig_expert_play, table2, fig_friction, smoke}
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPO=$REPO_ROOT
F=$REPO/findings/260907_final_dtg_s1_warm
cat=$1; run=$2; jid=$3; shift 3
warm="--warm"; extra=()
while [ $# -gt 0 ]; do case "$1" in --no-warm) warm="";; *) extra+=("$1");; esac; shift; done
rundir=$F/runs/$cat/$run
out=$REPO/logs_smoke/${run}_${jid}.out
[ -d "$rundir" ] || { echo "no run dir $rundir"; exit 1; }
[ -e "$out" ] || { echo "no job log $out"; exit 1; }
echo "=== gate $run ($jid) ==="
python3 $F/scripts/gate_check.py "$rundir" "$out" $warm "${extra[@]}" | tee $F/provenance/gate_${run}.txt | tail -1
grc=${PIPESTATUS[0]}
cp $REPO/logs_smoke/hydra/${run}_${jid}/.hydra/overrides.yaml $F/provenance/${run}.job${jid}.overrides.yaml 2>/dev/null || echo "(no hydra overrides dir for $run)"
cp "$out" $F/runs/$cat/$run/joblog.out 2>/dev/null
sj=$(sbatch --parsable --export=ALL,RUNS="$rundir",OUT=$F/scores/${cat}__${run}.json $F/scripts/score_runs.sbatch)
L=$REPO/logs_smoke; for i in $(seq 1 45); do [ -e $L/score_runs_$sj.out ] && grep -q 'DONE score_runs' $L/score_runs_$sj.out && break; sleep 20; done
row=$(grep -E ' n=|ERROR' $L/score_runs_$sj.out | head -1)
echo "$row"
printf "| %s | %s | %s | %s | gate=%s | %s |\n" "$(date +%m-%d\ %H:%M)" "$cat" "$run" "$jid" "$([ $grc -eq 0 ] && echo PASS || echo FAIL)" "$row" >> $F/provenance/results_ledger.md
