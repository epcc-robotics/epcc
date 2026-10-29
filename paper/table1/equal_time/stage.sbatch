#!/usr/bin/env bash
# Equal-time rescoring, one stage per job:   sbatch --job-name=eqt<N>_<name> stage.sbatch <stage>
#   table1     Table I at equal time            -> scores/table1_eqtime.json
#   fig3bdata  filter the Fig. 3b alpha dump    -> data/alpha_per_episode_tableI_eqtime.json
#   fig3a      Fig. 3a: regression at own length (must equal published scores + PNG), then 59.4 s
#   fig3b      Fig. 3b: regression on the published dump (must equal published PNG + values), then eq
#   summary    render/table1/table1_eqtime.md + render/summary.md
#   beforeafter  before/after tables and figures -> render/before_after/
#   sailk      SAIL best-k by TP at equal time (all k, r1) -> scores/sail_k_eqtime.json
# Logs: findings/261025_equal_time/logs/<job-name>_<jobid>.out. Writes only under 261025_equal_time.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPRO=$REPO_ROOT
cd "$REPRO" || exit 1
E=findings/261025_equal_time
S=$E/scripts
STAGE=${1:?stage}
echo "=== stage=$STAGE job=$SLURM_JOB_ID $(date -Is)"
fail=0
case "$STAGE" in
  table1)
    python -u $S/score_table1_eqtime.py || fail=1 ;;
  fig3bdata)
    python -u $S/make_fig3b_data_eqtime.py || fail=1 ;;
  fig3a)
    mkdir -p $E/scores/regress
    for m in published robot; do
      echo "######## Fig. 3a SUP_MODE=$m  regression (own length)"
      SUP_MODE=$m EQ_T=none OUT_DIR=$E/render/regress/fig3a/$m SCORES_JSON=$E/scores/regress/fig3a_$m.json \
        FIG_W=3.54 FIG_H=1.80 FIG_PT=8 TITLES=1 SUF=_named python -u $S/plot_fig3a_eqtime.py | tail -16 || fail=1
      python -u $S/compare_outputs.py json $E/scores/regress/fig3a_$m.json findings/261023_sup_robot/scores/friction_scores_$m.json || fail=1
      python -u $S/compare_outputs.py bytes $E/render/regress/fig3a/$m/friction_plain3way_1to1_named.png \
        findings/261023_sup_robot/fig3a/$m/friction_plain3way_1to1_named.png || fail=1
      echo "######## Fig. 3a SUP_MODE=$m  equal time 59.4 s"
      SUP_MODE=$m EQ_T=59.4 OUT_DIR=$E/render/fig3a/$m SCORES_JSON=$E/scores/fig3a_eqtime_$m.json \
        FIG_W=3.54 FIG_H=1.80 FIG_PT=8 TITLES=1 SUF=_named python -u $S/plot_fig3a_eqtime.py | tail -16 || fail=1
    done
    python -u $S/compare_outputs.py bytes findings/261023_sup_robot/fig3a/published/friction_plain3way_1to1_named.png \
      findings/260907_final_dtg_s1_warm/fig_friction/friction_plain3way_1to1_named.png || echo "  (note: 261023 published copy vs 260907 paper PNG)" ;;
  fig3b)
    PAPER="FIG_W=3.54 FIG_H=1.80 FIG_PT=8 RIGHT_RECT=0.94 CUBE_MAP=precision SUF=_prec"
    PUBDATA=$REPRO/findings/261022_fig3b_sail/data/alpha_per_episode_tableI.json
    for spec in tableI:full:paired4:0 tableI_own:full:own:1 tableI_suprobot:robot:paired4:0 \
                tableI_suprobot_own:robot:own:1 tableI_suprobot_own_noci:robot:own:0; do
      IFS=: read -r name sup agg err <<< "$spec"
      echo "######## Fig. 3b $name  regression (published dump)"
      env $PAPER SUP_MODE=$sup AGG=$agg ERR=$err PHASE_XLAB='motion phase' DATA=$PUBDATA \
        OUT_DIR=$E/render/regress/fig3b/$name python -u $S/render_fig3b_eqtime.py | tail -2 || fail=1
      for f in contact_split_3task_precision_values.json contact_split_3task_precision_1to1_prec.png; do
        python -u $S/compare_outputs.py bytes $E/render/regress/fig3b/$name/$f findings/261022_fig3b_sail/render/$name/$f || fail=1
      done
      echo "######## Fig. 3b $name  equal time"
      env $PAPER SUP_MODE=$sup AGG=$agg ERR=$err PHASE_XLAB='motion phase' \
        OUT_DIR=$E/render/fig3b/$name python -u $S/render_fig3b_eqtime.py || fail=1
    done ;;
  summary)
    python -u $S/render_summary_eqtime.py || fail=1 ;;
  beforeafter)
    python -u $S/render_before_after.py || fail=1 ;;
  sailk)
    python -u $S/score_sail_k_eqtime.py || fail=1 ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "=== DONE stage=$STAGE fail=$fail $(date -Is)"
exit $fail
