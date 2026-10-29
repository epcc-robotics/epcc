# paper/

Scripts that reproduce the paper's tables and figures from evaluation run
directories. They were moved here on 2026-10-27 from the research tree's
`findings/<campaign>/scripts/`. The Table I equal-time and Table II scorers read the
published runs from `EPCC_PUBLISHED`; the Fig. 3, Fig. 4 and scaling scripts still expect the
research layout of run directories and are being parametrized.

| Directory | Paper element | Origin campaign |
|---|---|---|
| `common/` | scoring (`score_runs.py`, `gate_check.py`), post-run step, video render | 260907_final_dtg_s1_warm |
| `table1/` | Table I EPCC and Play+Goal launchers, Table I index and scoring, SAIL row scorers | 260907_final_dtg_s1_warm |
| `table1/sup_robot/` | robot-only SuP rerun (Table I SuP row, Fig. 3a SuP curve) | 261023_sup_robot |
| `table1/equal_time/` | equal-time rescoring of Table I, plus Fig. 3a/3b equal-time variants | 261025_equal_time |
| `table2/` | Table II launcher and table | 260907_final_dtg_s1_warm |
| `fig3/` | Fig. 3a friction sweep plot, Fig. 3b contact and phase split | 260907_final_dtg_s1_warm |
| `fig3/fig3b_sail/` | Fig. 3b per-episode alpha dump and render | 261022_fig3b_sail |
| `fig4/` | Fig. 4 launcher, collection-pace audit (x axis) | 260907_final_dtg_s1_warm |
| `fig4/cold/` | Fig. 4 cold-predictor rerun: train, select, eval, plot | 261022_fig4_cold |
| `scaling/` | appendix data-scaling sweep: subsets, train, select, eval, plot | 261001_scaling |

Shared library code stays in `scripts/baseline/`: `park_metrics.py` (success and
time-to-completion metrics) and `paper_style.py` (plot style).
