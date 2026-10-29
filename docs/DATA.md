# Data layout (`STABLEWM_HOME`)

Every launcher reads `STABLEWM_HOME` (set in `cluster.env`) as the root of datasets,
checkpoints and caches. The layout below is what the paper cells in `config/eval/cells/paper/`
reference; the release archive (checkpoints and datasets, published separately) unpacks into it.
Training outputs go under `train/{wm,dtg,dp}/`; evaluation outputs stay in the repository under `results/`.

Checkpoints keep the trainer work-dir layout `<run>/models/<ckpt>.pt` with the sibling files the loaders read:
`<run>/normalizer_stats.npz` (DP, SkiP, SAIL, DTG) and `<run>/meta.json` (DTG).

| Path under `STABLEWM_HOME` | Used by (paper cells) |
|---|---|
| `checkpoints/agg_cubeM2_object.ckpt` | 7 cells |
| `checkpoints/agg_plainM2_object.ckpt` | 7 cells |
| `checkpoints/agg_slipM1_object.ckpt` | table1/slippery_pusht_epcc |
| `checkpoints/agg_slipM2_object.ckpt` | 6 cells |
| `checkpoints/cube_wm/lewm_pv_aw_cube_scratch_epoch_11_object.ckpt` | 7 cells |
| `checkpoints/dp/pusht_h16/models/model_final.pt` | 11 cells |
| `checkpoints/dp/pusht_h16/normalizer_stats.npz` | 11 cells |
| `checkpoints/dp/slippery_pusht_h16/models/model_final.pt` | table1/slippery_pusht_dp, table1/slippery_pusht_sup |
| `checkpoints/dp/slippery_pusht_h16/normalizer_stats.npz` | table1/slippery_pusht_dp, table1/slippery_pusht_sup |
| `checkpoints/dtg/cube_s1H84_lv30/meta.json` | 19 cells |
| `checkpoints/dtg/cube_s1H84_lv30/models/model_34000.pt` | 19 cells |
| `checkpoints/dtg/cube_s1H84_lv30/normalizer_stats.npz` | 19 cells |
| `checkpoints/dtg/pusht_s1H84_lv30/meta.json` | 19 cells |
| `checkpoints/dtg/pusht_s1H84_lv30/models/model_100000.pt` | 19 cells |
| `checkpoints/dtg/pusht_s1H84_lv30/normalizer_stats.npz` | 19 cells |
| `checkpoints/dtg/slippery_pusht_s1H84_lv30/meta.json` | 19 cells |
| `checkpoints/dtg/slippery_pusht_s1H84_lv30/models/model_24000.pt` | 19 cells |
| `checkpoints/dtg/slippery_pusht_s1H84_lv30/normalizer_stats.npz` | 19 cells |
| `checkpoints/f4cold/cube_D03_object.ckpt` | fig4/cube_D03 |
| `checkpoints/f4cold/cube_D075_object.ckpt` | fig4/cube_D075 |
| `checkpoints/f4cold/cube_D12_object.ckpt` | fig4/cube_D12 |
| `checkpoints/f4cold/plain_CTRL_object.ckpt` | fig4/pusht_CTRL |
| `checkpoints/f4cold/plain_D05_object.ckpt` | fig4/pusht_D05 |
| `checkpoints/f4cold/plain_D15_object.ckpt` | fig4/pusht_D15 |
| `checkpoints/f4cold/plain_D25_object.ckpt` | fig4/pusht_D25 |
| `checkpoints/f4cold/plain_DEMO_object.ckpt` | fig4/pusht_DEMO |
| `checkpoints/f4cold/slippery_CTRL_object.ckpt` | fig4/slippery_pusht_CTRL |
| `checkpoints/f4cold/slippery_D05_object.ckpt` | fig4/slippery_pusht_D05 |
| `checkpoints/f4cold/slippery_D15_object.ckpt` | fig4/slippery_pusht_D15 |
| `checkpoints/f4cold/slippery_D25_object.ckpt` | fig4/slippery_pusht_D25 |
| `checkpoints/f4cold/slippery_DEMO_object.ckpt` | fig4/slippery_pusht_DEMO |
| `checkpoints/fricd000_object.ckpt` | fig3a/epcc_d000 |
| `checkpoints/fricd010_object.ckpt` | fig3a/epcc_d010 |
| `checkpoints/fricd020_object.ckpt` | fig3a/epcc_d020 |
| `checkpoints/fricd030_object.ckpt` | fig3a/epcc_d030 |
| `checkpoints/fricd040_object.ckpt` | fig3a/epcc_d040 |
| `checkpoints/fricd050_object.ckpt` | fig3a/epcc_d050 |
| `checkpoints/fricd060_object.ckpt` | fig3a/epcc_d060 |
| `checkpoints/fricd070_object.ckpt` | fig3a/epcc_d070 |
| `checkpoints/fricd080_object.ckpt` | fig3a/epcc_d080 |
| `checkpoints/friction/d000/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd000/42/meta.json` | fig3a/epcc_d000 |
| `checkpoints/friction/d000/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd000/42/models/selected_val.pt` | fig3a/epcc_d000 |
| `checkpoints/friction/d000/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd000/42/normalizer_stats.npz` | fig3a/epcc_d000 |
| `checkpoints/friction/d010/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd010/42/meta.json` | fig3a/epcc_d010 |
| `checkpoints/friction/d010/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd010/42/models/selected_val.pt` | fig3a/epcc_d010 |
| `checkpoints/friction/d010/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd010/42/normalizer_stats.npz` | fig3a/epcc_d010 |
| `checkpoints/friction/d020/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd020/42/meta.json` | fig3a/epcc_d020 |
| `checkpoints/friction/d020/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd020/42/models/selected_val.pt` | fig3a/epcc_d020 |
| `checkpoints/friction/d020/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd020/42/normalizer_stats.npz` | fig3a/epcc_d020 |
| `checkpoints/friction/d030/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd030/42/meta.json` | fig3a/epcc_d030 |
| `checkpoints/friction/d030/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd030/42/models/selected_val.pt` | fig3a/epcc_d030 |
| `checkpoints/friction/d030/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd030/42/normalizer_stats.npz` | fig3a/epcc_d030 |
| `checkpoints/friction/d040/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd040/42/meta.json` | fig3a/epcc_d040 |
| `checkpoints/friction/d040/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd040/42/models/selected_val.pt` | fig3a/epcc_d040 |
| `checkpoints/friction/d040/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd040/42/normalizer_stats.npz` | fig3a/epcc_d040 |
| `checkpoints/friction/d050/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd050/42/meta.json` | fig3a/epcc_d050 |
| `checkpoints/friction/d050/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd050/42/models/selected_val.pt` | fig3a/epcc_d050 |
| `checkpoints/friction/d050/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd050/42/normalizer_stats.npz` | fig3a/epcc_d050 |
| `checkpoints/friction/d060/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd060/42/meta.json` | fig3a/epcc_d060 |
| `checkpoints/friction/d060/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd060/42/models/selected_val.pt` | fig3a/epcc_d060 |
| `checkpoints/friction/d060/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd060/42/normalizer_stats.npz` | fig3a/epcc_d060 |
| `checkpoints/friction/d070/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd070/42/meta.json` | fig3a/epcc_d070 |
| `checkpoints/friction/d070/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd070/42/models/selected_val.pt` | fig3a/epcc_d070 |
| `checkpoints/friction/d070/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd070/42/normalizer_stats.npz` | fig3a/epcc_d070 |
| `checkpoints/friction/d080/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd080/42/meta.json` | fig3a/epcc_d080 |
| `checkpoints/friction/d080/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd080/42/models/selected_val.pt` | fig3a/epcc_d080 |
| `checkpoints/friction/d080/diffusion_cd_s1H84/pusht_latent_velloss/A_mm_s1H84_plaindemo_fricd080/42/normalizer_stats.npz` | fig3a/epcc_d080 |
| `checkpoints/pusht_wm/lewm_object.ckpt` | 25 cells |
| `checkpoints/sail/pusht_reached/models/model_final.pt` | 13 cells |
| `checkpoints/sail/pusht_reached/normalizer_stats.npz` | 13 cells |
| `checkpoints/sail/slippery_pusht_reached/models/model_final.pt` | 4 cells |
| `checkpoints/sail/slippery_pusht_reached/normalizer_stats.npz` | 4 cells |
| `checkpoints/scaling/cube_f128_object.ckpt` | scaling/cube_f128 |
| `checkpoints/scaling/cube_f16_object.ckpt` | scaling/cube_f16 |
| `checkpoints/scaling/cube_f1_object.ckpt` | scaling/cube_f1 |
| `checkpoints/scaling/cube_f256_object.ckpt` | scaling/cube_f256 |
| `checkpoints/scaling/cube_f32_object.ckpt` | scaling/cube_f32 |
| `checkpoints/scaling/cube_f4_object.ckpt` | scaling/cube_f4 |
| `checkpoints/scaling/cube_f64_object.ckpt` | scaling/cube_f64 |
| `checkpoints/scaling/plain_f128_object.ckpt` | scaling/pusht_f128 |
| `checkpoints/scaling/plain_f16_object.ckpt` | scaling/pusht_f16 |
| `checkpoints/scaling/plain_f1_object.ckpt` | scaling/pusht_f1 |
| `checkpoints/scaling/plain_f256_object.ckpt` | scaling/pusht_f256 |
| `checkpoints/scaling/plain_f32_object.ckpt` | scaling/pusht_f32 |
| `checkpoints/scaling/plain_f4_object.ckpt` | scaling/pusht_f4 |
| `checkpoints/scaling/plain_f64_object.ckpt` | scaling/pusht_f64 |
| `checkpoints/scaling/slippery_f128_object.ckpt` | scaling/slippery_pusht_f128 |
| `checkpoints/scaling/slippery_f16_object.ckpt` | scaling/slippery_pusht_f16 |
| `checkpoints/scaling/slippery_f1_object.ckpt` | scaling/slippery_pusht_f1 |
| `checkpoints/scaling/slippery_f256_object.ckpt` | scaling/slippery_pusht_f256 |
| `checkpoints/scaling/slippery_f32_object.ckpt` | scaling/slippery_pusht_f32 |
| `checkpoints/scaling/slippery_f4_object.ckpt` | scaling/slippery_pusht_f4 |
| `checkpoints/scaling/slippery_f64_object.ckpt` | scaling/slippery_pusht_f64 |
| `checkpoints/scaling/smoke_cube_f256_object.ckpt` | scaling/cube_f256_smoke |
| `checkpoints/scaling/smoke_plain_f256_object.ckpt` | scaling/pusht_f256_smoke |
| `checkpoints/scaling/smoke_slippery_f256_object.ckpt` | scaling/slippery_pusht_f256_smoke |
| `checkpoints/skip/cube_h16/models/model_final.pt` | table1/cube_skip |
| `checkpoints/skip/cube_h16/normalizer_stats.npz` | table1/cube_skip |
| `checkpoints/skip/pusht_h16/models/model_final.pt` | table1/pusht_skip |
| `checkpoints/skip/pusht_h16/normalizer_stats.npz` | table1/pusht_skip |
| `checkpoints/skip/slippery_pusht_h16/models/model_final.pt` | table1/slippery_pusht_skip |
| `checkpoints/skip/slippery_pusht_h16/normalizer_stats.npz` | table1/slippery_pusht_skip |
| `checkpoints/slippery_pusht/lewm_object.ckpt` | 7 cells |
| `checkpoints/wm_demoonly/cube_demo_object.ckpt` | fig4/cube_DEMO |
| `checkpoints/wm_playsub/cube_object.ckpt` | fig4/cube_CTRL |
| `cube/dp/models/model_90000.pt` | table1/cube_dp, table1/cube_sup |
| `cube/dp/normalizer_stats.npz` | table1/cube_dp, table1/cube_sup |
| `cube/invdyn/best.pt` | 4 cells |
| `cube/latents/cube_multiview_diffusion_aw_c0.3_postpad_fullscale.h5` | 19 cells |
| `cube/sail/awe/cube_reached_precision_c03.h5` | 4 cells |
| `cube/sail/dp_reached/models/model_final.pt` | 4 cells |
| `cube/sail/dp_reached/normalizer_stats.npz` | 4 cells |
| `cube/sup/adm_A.pt` | table1/cube_sup |
| `cube/sup/adm_B.pt` | 4 cells |
| `cube/sup/iql_full_h128.pt` | 4 cells |
| `cube/sup/iqlak_robot_park_paper.pt` | table1/cube_sup |
| `datasets/agg_cube_D_c0.3.h5 (dataset_stats)` | fig4/cube_D03 |
| `datasets/agg_cube_D_c0.75.h5 (dataset_stats)` | fig4/cube_D075 |
| `datasets/agg_cube_D_c1.2.h5 (dataset_stats)` | fig4/cube_D12 |
| `datasets/agg_cube_main2.h5 (dataset_stats)` | 7 cells |
| `datasets/agg_plain_D_c0.5.h5 (dataset_stats)` | fig4/pusht_D05 |
| `datasets/agg_plain_D_c1.5.h5 (dataset_stats)` | fig4/pusht_D15 |
| `datasets/agg_plain_D_c2.5.h5 (dataset_stats)` | fig4/pusht_D25 |
| `datasets/agg_plain_main2.h5 (dataset_stats)` | 7 cells |
| `datasets/agg_slippery_D_c0.5.h5 (dataset_stats)` | fig4/slippery_pusht_D05 |
| `datasets/agg_slippery_D_c1.5.h5 (dataset_stats)` | fig4/slippery_pusht_D15 |
| `datasets/agg_slippery_D_c2.5.h5 (dataset_stats)` | fig4/slippery_pusht_D25 |
| `datasets/agg_slippery_main1.h5 (dataset_stats)` | table1/slippery_pusht_epcc |
| `datasets/agg_slippery_main2.h5 (dataset_stats)` | 6 cells |
| `datasets/cube_multiview_WM.h5 (dataset_stats)` | 7 cells |
| `datasets/cube_multiview_WM_sub256k.h5 (dataset_stats)` | fig4/cube_CTRL |
| `datasets/cube_multiview_diffusion_c0.3_postpad_fullscale.h5` | 7 cells |
| `datasets/cube_multiview_diffusion_c0.3_postpad_fullscale.h5 (dataset_stats)` | fig4/cube_DEMO |
| `datasets/cube_multiview_diffusion_c0.3_postpad_fullscale.h5 (eval dataset)` | 27 cells |
| `datasets/expert_slow_025_n200_centered_v2.h5 (eval dataset)` | 27 cells |
| `datasets/expert_slow_025_n200_centered_v2_park.h5` | 16 cells |
| `datasets/expert_slow_025_n200_centered_v2_park.h5 (dataset_stats)` | fig4/pusht_DEMO |
| `datasets/pusht/data/pusht_expert_train.h5 (dataset_stats)` | 32 cells |
| `datasets/pusht_expert_train_sub83k.h5 (dataset_stats)` | fig4/pusht_CTRL |
| `datasets/scaling/cube/cube_main2_f1.h5 (dataset_stats)` | scaling/cube_f1 |
| `datasets/scaling/cube/cube_main2_f128.h5 (dataset_stats)` | scaling/cube_f128 |
| `datasets/scaling/cube/cube_main2_f16.h5 (dataset_stats)` | scaling/cube_f16 |
| `datasets/scaling/cube/cube_main2_f256.h5 (dataset_stats)` | scaling/cube_f256, scaling/cube_f256_smoke |
| `datasets/scaling/cube/cube_main2_f32.h5 (dataset_stats)` | scaling/cube_f32 |
| `datasets/scaling/cube/cube_main2_f4.h5 (dataset_stats)` | scaling/cube_f4 |
| `datasets/scaling/cube/cube_main2_f64.h5 (dataset_stats)` | scaling/cube_f64 |
| `datasets/scaling/plain/plain_main2_f1.h5 (dataset_stats)` | scaling/pusht_f1 |
| `datasets/scaling/plain/plain_main2_f128.h5 (dataset_stats)` | scaling/pusht_f128 |
| `datasets/scaling/plain/plain_main2_f16.h5 (dataset_stats)` | scaling/pusht_f16 |
| `datasets/scaling/plain/plain_main2_f256.h5 (dataset_stats)` | scaling/pusht_f256, scaling/pusht_f256_smoke |
| `datasets/scaling/plain/plain_main2_f32.h5 (dataset_stats)` | scaling/pusht_f32 |
| `datasets/scaling/plain/plain_main2_f4.h5 (dataset_stats)` | scaling/pusht_f4 |
| `datasets/scaling/plain/plain_main2_f64.h5 (dataset_stats)` | scaling/pusht_f64 |
| `datasets/scaling/slippery/slippery_main2_f1.h5 (dataset_stats)` | scaling/slippery_pusht_f1 |
| `datasets/scaling/slippery/slippery_main2_f128.h5 (dataset_stats)` | scaling/slippery_pusht_f128 |
| `datasets/scaling/slippery/slippery_main2_f16.h5 (dataset_stats)` | scaling/slippery_pusht_f16 |
| `datasets/scaling/slippery/slippery_main2_f256.h5 (dataset_stats)` | scaling/slippery_pusht_f256, scaling/slippery_pusht_f256_smoke |
| `datasets/scaling/slippery/slippery_main2_f32.h5 (dataset_stats)` | scaling/slippery_pusht_f32 |
| `datasets/scaling/slippery/slippery_main2_f4.h5 (dataset_stats)` | scaling/slippery_pusht_f4 |
| `datasets/scaling/slippery/slippery_main2_f64.h5 (dataset_stats)` | scaling/slippery_pusht_f64 |
| `datasets/slippery_pusht/data/expert_slow_050_n200_d035.h5 (eval dataset)` | 54 cells |
| `datasets/slippery_pusht/data/slippery_pusht_wm_corpus.h5 (dataset_stats)` | 10 cells |
| `datasets/slippery_pusht/data/slippery_pusht_wm_corpus_v1_sub101k.h5 (dataset_stats)` | fig4/slippery_pusht_CTRL |
| `pusht/latents/expert_slow_025_n200_centered_v2_latents_lewm_fp32.h5` | 19 cells |
| `pusht/sail/awe/pusht_reached_precision_park.h5` | 13 cells |
| `pusht/sup/adm_A.pt` | 10 cells |
| `pusht/sup/adm_B.pt` | 4 cells |
| `pusht/sup/iql_full_h128.pt` | 4 cells |
| `pusht/sup/iqlak_robot_park_paper.pt` | 10 cells |
| `pusht_invdyn/best.pt` | 4 cells |
| `slippery_pusht/data/expert_slow_050_n200_d035_park.h5` | 7 cells |
| `slippery_pusht/data/latent/expert_slow_050_n200_d035_park_latents_lewm_base.h5` | 19 cells |
| `slippery_pusht/data/latent/expert_slow_050_n200_d035_park_latents_lewm_fricd000.h5` | fig3a/epcc_d000 |
| `slippery_pusht/data/latent/expert_slow_050_n200_d035_park_latents_lewm_fricd010.h5` | fig3a/epcc_d010 |
| `slippery_pusht/data/latent/expert_slow_050_n200_d035_park_latents_lewm_fricd020.h5` | fig3a/epcc_d020 |
| `slippery_pusht/data/latent/expert_slow_050_n200_d035_park_latents_lewm_fricd030.h5` | fig3a/epcc_d030 |
| `slippery_pusht/data/latent/expert_slow_050_n200_d035_park_latents_lewm_fricd040.h5` | fig3a/epcc_d040 |
| `slippery_pusht/data/latent/expert_slow_050_n200_d035_park_latents_lewm_fricd050.h5` | fig3a/epcc_d050 |
| `slippery_pusht/data/latent/expert_slow_050_n200_d035_park_latents_lewm_fricd060.h5` | fig3a/epcc_d060 |
| `slippery_pusht/data/latent/expert_slow_050_n200_d035_park_latents_lewm_fricd070.h5` | fig3a/epcc_d070 |
| `slippery_pusht/data/latent/expert_slow_050_n200_d035_park_latents_lewm_fricd080.h5` | fig3a/epcc_d080 |
| `slippery_pusht/invdyn_base/best.pt` | 4 cells |
| `slippery_pusht/sail/awe/slippery_reached_precision_park.h5` | 4 cells |
| `slippery_pusht/sup/adm_A.pt` | table1/slippery_pusht_sup |
| `slippery_pusht/sup/adm_B.pt` | 4 cells |
| `slippery_pusht/sup/iql_full_h128.pt` | 4 cells |
| `slippery_pusht/sup/iqlak_robot_park_paper.pt` | table1/slippery_pusht_sup |

Derived names: `checkpoints/<policy_config_group>_object.ckpt` is the world model of a cell,
`datasets/<dataset_stats>.h5` the dataset whose statistics normalize it, and `datasets/<eval dataset>.h5`
the episode set the cell is evaluated on. Training recipes (`config/train/`, `latent_modules/diffusion_cd/configs/`,
`scripts/data/config/`) reference the same tree.

Data collection (`scripts/data/`, `scripts/collect_*`) is provided as run on the research cluster; the collected
datasets are released as files and bit-identical regeneration is not guaranteed.
