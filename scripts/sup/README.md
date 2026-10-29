# SuP — SpeedUpPatch scheduler for a frozen diffusion policy

Accelerate a **frozen** diffusion policy by executing its action chunk at a coarser rate `k`,
with a world-model safety gate deciding how large `k` may be at each state
(SpeedUpPatch, arXiv:2603.20658). **Nothing here retrains the policy.**

Two channels give a controlled comparison:

| channel | ADM track | state the WM sees | meaning |
|---|---|---|---|
| `robot` | A | effector/agent only | **faithful SuP** (as in the paper) |
| `full`  | B | + object (block) | **ours** (object-aware) |

---

## 1. Requirements

### Environment
```bash
cd <repo root>
REPO_ROOT=$PWD; source scripts/env.sh   # venv, STABLEWM_HOME, PYTHONPATH from cluster.env
export PYTORCH_ALLOC_CONF=expandable_segments:True
```
One GPU. An ADM trains in ~5–20 min, the IQL in ~3 min, a 50-env eval takes 5–45 min.

### Inputs you must already have

| task | frozen base policy (never retrained) | WM corpus (ADM training) | demo set (D′ synthesis) |
|---|---|---|---|
| `slippery` | `slippery_pusht/policy/dp_d035_S050_Ta16_Tp32/checkpoints/latest.ckpt` | `slippery_pusht/data/slippery_pusht_wm_corpus.h5` | `slippery_pusht/data/expert_slow_050_n200_d035.h5` |
| `pusht` | CleanDiffuser DP (see `config/eval/ws_reproduce/pusht_sup_*.yaml`) | `pusht/data/pusht_expert_train.h5` | `pusht/data/expert_slow_025_n200.h5` |
| `cube` | CleanDiffuser DP (see `config/eval/ws_reproduce/cube_sup_*.yaml`) | `cube/data/cube_multiview_WM.h5` | `cube/data/cube_multiview_diffusion_c03.h5` |

Paths are relative to `$STABLEWM_HOME`. The base policy is **shared with the non-SuP baselines**
— it stays under `<task>/policy/`, not under `sup/`.

### Outputs — the asset contract
Every SuP asset for a task lives in **one** directory:
```
$STABLEWM_HOME/<task>/sup/          # <task> = slippery_pusht | pusht | cube
    adm_A.pt   adm_B.pt             # dynamics   (robot / +object)
    adm_A_val.json  adm_B_val.json  # per-horizon validation error
    iql_robot_dataset.npz  iql_full_dataset.npz   # CMDP dataset D'
    iql_robot.pt   iql_full.pt      # scheduler
```
Scripts resolve this directory from `--task` (override with `--out_dir` / `--ckpt_dir`);
configs reference it as `${cache_dir}/<task>/sup/...`.

---

## 2. Pipeline

```
   1. train ADM               2. synthesize D' + train IQL          3. evaluate
 train_adm.py  ──────────►  build_iql_dataset.py ─► train_iql.py ─►  eval_unified.py
 adm_{A,B}.pt                iql_{ch}_dataset.npz    iql_{ch}.pt      (policy/sup_*.py)
```

### 1. Train the ADM (any-step world model)
One trainer, `--task {slippery,pusht,cube}` × `--track {A,B}`:
```bash
python scripts/sup/train_adm.py --task slippery --track A --epochs 25
python scripts/sup/train_adm.py --task slippery --track B --epochs 40
python scripts/sup/train_adm.py --task pusht    --track A --epochs 18
python scripts/sup/train_adm.py --task pusht    --track B --epochs 30
python scripts/sup/train_adm.py --task cube     --track A --epochs 20
python scripts/sup/train_adm.py --task cube     --track B --epochs 35
```
Sanity-check `adm_*_val.json`: the per-horizon error should stay roughly **flat in k**. That
flatness is what makes `E(s,k)` usable at large `k` — an any-step model never feeds its own
prediction back, so error does not compound.

### 2. Train the IQL scheduler (per task × channel)
```bash
for task in slippery pusht cube; do
  d=$STABLEWM_HOME/$([ $task = slippery ] && echo slippery_pusht || echo $task)/sup
  for ch in robot full; do
    python scripts/sup/build_iql_dataset.py --task $task --channel $ch     # --chunk C
    python scripts/sup/train_iql.py --dataset $d/iql_${ch}_dataset.npz \
                                    --out     $d/iql_${ch}.pt --eps_q 0.4
  done
done
```
`build_iql_dataset.py` rolls the ADM twice per (state, k) — once on the fine chunk, once on the
k-merged chunk — and records the safety cost `E(s,k)`. The CMDP reward is `r' = k` when
`E <= eps`, else `-OMEGA` (600, per SuP Prop. 4.1). **`--chunk` must match the chunk length of
the policy you will gate** (Known issues #1).

### 3. Evaluate
One entry point; `score_mode` selects what you are running:

| what | `policy.sup.score_mode` | extra |
|---|---|---|
| **fixed-k** (SuP − scheduler) | `none` | `policy.sup.fixed_k=K` (`K=1` ⇒ bit-exact base DP) |
| greedy WM gate (SuP "MPC" baseline) | `state_adm` | `eps` |
| **IQL scheduler** (full SuP) | `state_adm_iql` | `iql_path` |

```bash
# SLIPPERY — robomimic DP-repo backbone (absolute waypoints ⇒ decimate)
EVAL_CONFIG_DIR=../../config/eval/baseline_corl_v2 \
python scripts/baseline/eval_unified.py --config-name slippery_pusht_sup_state_gate \
  task.num_eval=50 policy.sup.score_mode=state_adm_iql policy.sup.channel=full

# fixed-k sweep (no scheduler)
EVAL_CONFIG_DIR=../../config/eval/baseline_corl_v2 \
python scripts/baseline/eval_unified.py --config-name slippery_pusht_sup_dp_repo \
  task.num_eval=50 policy.sup.score_mode=none policy.sup.fixed_k=3

# CUBE / PUSHT — CleanDiffuser DP backbone (relative deltas ⇒ accumulate)
EVAL_CONFIG_DIR=../../config/eval/ws_reproduce \
python scripts/baseline/eval_unified.py --config-name cube_sup_state_gate \
  task.num_eval=50 policy.sup.score_mode=state_adm_iql policy.sup.channel=robot \
  policy.sup.adm_path=$STABLEWM_HOME/cube/sup/adm_A.pt \
  policy.sup.iql_path=$STABLEWM_HOME/cube/sup/iql_robot.pt
```
Useful overrides: `+eval.save_traj=true` dumps the per-step pymunk state to `rollout_traj.npz`
— enough to re-score any success criterion offline without re-simulating. Every gated run also
writes `sup_decisions.jsonl` with the per-replan, per-env chosen `k`.

---

## 3. Evaluation condition (official)

**PushT-family evals run walls-off**: `world.escapable_block: true`, already set in the SuP
configs. The stock env has four static walls that catch a diverging T; with them removed the T
can actually leave the arena, so divergence caused by aggressive downsampling shows up as
failure instead of being silently contained. Only the block is freed — the agent is a KINEMATIC
body driven by PD and was never affected by the walls. (`cube` is a MuJoCo task; the flag does
not apply.)

⚠️ Numbers measured **before** this switch (the walls-on results in §6) are not comparable to
new runs. Note also that the harness's `success_rate` is *ever-reached* — the block touching the
goal pose once latches success — which hides late divergence. Prefer a reached-**and-held**
criterion when reporting walls-off results.

---

## 4. Files

| file | role |
|---|---|
| `scripts/sup/adm_model.py` | ADM definition (GRU reads state, then actions; predicts Δstate at any horizon) |
| `scripts/sup/train_adm.py` | ① trains the ADM — task/track registry |
| `scripts/sup/build_iql_dataset.py` | ② rolls the ADM to score `E(s,k)`, emits the CMDP dataset `D'` |
| `scripts/sup/train_iql.py` | ② discrete-action IQL (expectile V + twin Q) |
| `policy/sup_utils.py` | pure helpers: chunk downsampling, per-env k choice |
| `policy/sup_gated.py` | ③ CleanDiffuser DP wrapper (cube, pusht) |
| `policy/sup_gated_dp_repo.py` | ③ robomimic DP-repo wrapper (slippery; absolute waypoints) |
| `policy/sup_state_gate.py` | ③ online gate: env state → ADM rollout → `E(s,k)` / `Q(s,k)` |
| `config/eval/{ws_reproduce,baseline_corl_v2}/*_sup_*.yaml` | eval recipes |

### The two downsampling conventions
* **accumulate** (paper Eq.2 Δ) — sum `k` consecutive *relative* deltas. CleanDiffuser DP (cube, pusht).
* **decimate** (paper Eq.2 Abs) — keep every k-th *absolute* waypoint. DP-repo (slippery).

They are equivalent: decimating absolute waypoints == accumulating the deltas between the kept
waypoints. `sup_gated_dp_repo.py` bridges them with `a_fine[t] = (w_t − w_{t−1}) / waypoint_scale`,
so a single ADM (trained on deltas) serves both backbones.

---

## 5. Known issues

1. **Chunk-length mismatch (MEDIUM).** The gate feeds the IQL `s = [env_state | flattened chunk]`.
   If the eval chunk `T*adim` differs from the IQL's training `C*adim` (`C=15`), the tail is
   truncated and a one-time WARN is printed. Live on the **DP-repo slippery** path
   (`chunk_size=16` → 39 vs trained 37); cube/pusht match at 15. The retained prefix is still
   correctly aligned — only the chunk's last action pair is dropped — so such runs are usable,
   but the scheduler decides from 15 of 16 actions. *Fix:* `build_iql_dataset.py --chunk 16`,
   then retrain the IQL.
2. **`accumulate` clipping (MEDIUM).** Merging *sums* `k` deltas, so a merged action can be up to
   `k×` a single one — **even when every original action is inside the bound.** Measured on the
   demo sets: pusht `k=6` → 4.0% of merged actions exceed ±1 (max 3.5); cube `k=4` → 53% exceed
   (max 1.2). The gate clips the merged action to `sat_limit` (1.0) while scoring `E(s,k)`, and
   so does `build_iql_dataset.py` (`CLIP=1.0`), but the **executed** action is *not* clipped
   (`sup_utils.downsample_actions` returns the raw sum). Consequences: for **cube** the env
   itself clips at ±1, so clipping while scoring is *correct* — it models real saturation. For
   **pusht** the env applies no clip, so the gate scores a weaker merged action than it will
   actually execute, `E(s,k)` comes out too large, and the gate is **needlessly conservative**.
   It biases both channels equally, so robot-vs-full comparisons still hold. The DP-repo path
   disables clipping (`clip=1e9`). *Related:* cube's action dim 4 is a 3-valued gripper command
   that `accumulate` also sums.
3. **Broad `except` → k=1 (LOW).** Gate build/scoring failures fall back to k=1 with a single
   printed warning. Safe, but a genuine ADM regression then looks like "the gate does nothing" —
   check for that warning before trusting a run that came out k=1-heavy.

---

## 6. Results so far (n=50, **walls-on** — superseded by §3)

Best fixed `k` beat every learned scheduler on all three tasks, and the object channel did not
improve the deployed scheduler:

| task | best fixed-k | robot-IQL (faithful SuP) | full-IQL (ours) |
|---|---|---|---|
| cube | 98% / 75 (k=2) | 96% / 131 | 94% / 130 |
| pusht | 100% / 95 (k=3) | 92% / 224 | 92% / 216 |
| slippery | 100% / 67 (k=2) | 90% / 106 | 84% / 113 |

(SR% / TTC steps.) `E(s,k)` is an excellent *aggregate* predictor of degradation but noisy
*per state*, which makes the schedulers over-conservative. Full analysis and figures live in the
knowledge base: `latent_mpcc_kb/{260716_speedup_patch_setup,260721_sup_speedup_gate}/`.
