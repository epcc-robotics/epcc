#!/usr/bin/env bash
# EXPERIMENT 6 (pushT cells) -- single-view WM trained on DEMOS ONLY.
#
# One file, two cells, selected by TASK=:
#
#   TASK=slippery   base cfg lewm_slippery_pusht_v2ft   (already has
#                   finetune_from=<champion base WM> + freeze_encoder=true)
#                   play corpus_v2 1,803,139 frames  ->  demos 81,310  (22x)
#
#   TASK=plain      base cfg lewm  (+freeze_encoder / +finetune_from added)
#                   play pusht_expert_train 2,336,736  ->  demos ~82,842  (28x)
#
# Encoder + projector are warm-started from the champion and FROZEN (plan
# decision 4): a demo-only encoder would define a different 192-d space and
# every cost weight / nu_scale / action_clamp is tuned in the champion's. Only
# predictor / action_encoder / pred_proj learn, and they learn from demos alone
# -- which is the thing under test.
#
# PARK demo sets, matching cube. Rationale (decided 2026-07-31):
#   - cube's only fullscale demo set IS post-padded (`..._postpad_fullscale`),
#     and it is what the cube DTG trains on AND what cube eval draws inits from.
#     There is no non-padded cube option that keeps the cube pipeline coherent.
#   - so a non-park pushT would make the three exp-6 cells measure different
#     things, and would also put the pushT WM on a different demo definition
#     than its own DTG (which trains on the park generation).
# An earlier revision used the non-park sets, on the reasoning that a park tail
# "inflates stationary transitions". That reasoning was weak: park_action
# (scripts/data/splice_park_tail_pusht.py:46-60, park_mode
# "radial_retreat_demo_pace") drives the agent AWAY from the block at demo pace
# and only holds once the retreat endpoint is reached — i.e. mostly
# agent-moves-block-stays transitions, a legitimate class, not dead frames.
#
# 🔴 KNOWN COUPLING -- read before evaluating any cell produced here.
# train.py builds the action normalizer as a z-score over the TRAINING set
# (utils.py:16-28) and does not store it in the model. Demo actions are far
# narrower than play actions (slippery |a| mean 0.085 vs 0.231; plain 0.059 vs
# 0.243), so a demo-trained WM lives in different normalized action units than
# the champion. At eval, `dataset.stats` and `solver.inner_solver.action_clamp`
# MUST be recomputed against the demo set, or CEM will propose actions in the
# wrong units and the cell will fail for a reason that has nothing to do with
# the hypothesis. This does not affect training -- only the eval config.
#
#   sbatch --export=TASK=slippery scripts/experiments/train/train_wm_demoonly_pusht.sbatch
#   sbatch --export=TASK=plain    scripts/experiments/train/train_wm_demoonly_pusht.sbatch
#
#SBATCH --job-name=wm_demoonly
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:TERM@300
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPO=$REPO_ROOT
H=$STABLEWM_HOME

TASK="${TASK:?set TASK=slippery|plain via --export}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"

# SMOKE=1 — see the note in train_perview_cube_demoonly.sbatch. Validates config
# composition (the two branches below need DIFFERENT override syntax), dataset
# resolution, the finetune load, the freeze, and one train+val step in minutes.
# Output namespaced so it cannot land on a real cell's checkpoints.
SMOKE_ARGS=()
SMOKE_TAG=""
if [ "${SMOKE:-0}" = "1" ]; then
    MAX_EPOCHS=1
    SMOKE_ARGS=(+trainer.limit_train_batches=5 +trainer.limit_val_batches=2)
    SMOKE_TAG="_smoke"
    echo "[SMOKE] 1 epoch / 5 train batches / 2 val batches"
fi

case "$TASK" in
  slippery)
    CFG=lewm_slippery_pusht_v2ft          # already: finetune_from champion + freeze_encoder
    # DS_REL is overridable so the exp-6 CONTROL cell can reuse this launcher:
    # play cut down to the demo frame budget, same recipe, only the data differs.
    DS_REL="${DS_REL:-slippery_pusht/data/expert_slow_050_n200_d035_park}"
    SRC="$H/datasets/${DS_REL}.h5"
    FT_CKPT="$H/checkpoints/slippery_pusht/lewm/lewm_object.ckpt"
    OUT_NAME="${OUT_NAME:-lewm_slippery_demoonly${SMOKE_TAG}}"
    # v2ft already declares finetune_from / freeze_encoder / data.dataset.cache_dir,
    # so these are plain overrides.
    EXTRA=(finetune_from="$FT_CKPT" freeze_encoder=true
           data.dataset.cache_dir="__LOCAL_BASE__")
    ;;
  plain)
    CFG=lewm                               # plain base cfg: add both keys with +
    DS_REL="${DS_REL:-expert_slow_025_n200_centered_v2_park}"
    SRC="$H/datasets/${DS_REL}.h5"
    FT_CKPT="$H/checkpoints/pusht_wm/lewm_object.ckpt"
    OUT_NAME="${OUT_NAME:-lewm_pusht_demoonly${SMOKE_TAG}}"
    # lewm.yaml + data/pusht.yaml declare NONE of these keys, and Hydra runs in
    # struct mode: a plain `key=value` override on a key that does not exist is
    # a ConfigAttributeError at composition time, before anything runs. All
    # three need the `+` (append) form here — unlike the slippery branch, where
    # v2ft already declares them.
    EXTRA=(+finetune_from="$FT_CKPT" +freeze_encoder=true
           +data.dataset.cache_dir="__LOCAL_BASE__")
    ;;
  *) echo "unknown TASK=$TASK (want slippery|plain)"; exit 2 ;;
esac

for f in "$SRC" "$FT_CKPT"; do
    [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done

# fail fast on schema before staging (train.py wants pixels/action/proprio/state)
python -u - "$SRC" <<'PY' || exit 1
import sys, hdf5plugin, h5py  # noqa: F401
need = ["pixels", "action", "proprio", "state", "ep_len", "ep_offset"]
with h5py.File(sys.argv[1], "r") as f:
    ks = set(f.keys())
    print(f"[schema] {sys.argv[1]}")
    print(f"[schema] frames={f['pixels'].shape[0]:,}  episodes={f['ep_len'].shape[0]:,}")
    miss = [k for k in need if k not in ks]
    for k in need:
        print(f"    {'OK  ' if k in ks else 'MISS'}  {k}")
    if miss:
        print(f"[schema] MISSING {miss}"); sys.exit(1)
    print("[schema] OK")
PY

# stage on node-local /tmp (Lustre random-read bottleneck, same as the
# slippery finetune launcher submit_finetune_v2.sbatch)
LOCAL_BASE="/tmp/$USER/stable-wm-${SLURM_JOB_ID:-local_$$}"
LOCAL_DATA="$LOCAL_BASE/datasets/$(dirname "$DS_REL")"
SUBDIR="${OUT_NAME}_${SLURM_JOB_ID:-local}"
mkdir -p "$LOCAL_DATA" "$LOCAL_BASE/$SUBDIR"
echo "[staging] $(du -h "$SRC" | cut -f1) -> $LOCAL_DATA"
cp "$SRC" "$LOCAL_DATA/" || exit 1

# EXTRA was built before LOCAL_BASE existed — fill the placeholder now.
EXTRA=("${EXTRA[@]/__LOCAL_BASE__/$LOCAL_BASE}")

CKPT_PERSIST="${CKPT_PERSIST:-$H/checkpoints/wm_demoonly/${OUT_NAME}}"
on_exit() {
    mkdir -p "$CKPT_PERSIST"
    cp -v "$LOCAL_BASE/$SUBDIR"/${OUT_NAME}* "$CKPT_PERSIST/" 2>/dev/null || true
    cp -v "$LOCAL_BASE/$SUBDIR/config.yaml" "$CKPT_PERSIST/" 2>/dev/null || true
    cp -v "$LOCAL_BASE/$SUBDIR"/*.ckpt "$CKPT_PERSIST/" 2>/dev/null || true
    ls -lh "$CKPT_PERSIST/" 2>/dev/null || true
    rm -rf "$LOCAL_BASE"
}
# EXIT alone is NOT enough. On walltime expiry SLURM sends SIGTERM; bash with no
# TERM trap is killed BY the signal and the EXIT trap never runs -- every
# checkpoint still sitting on node-local /tmp would be lost, along with the whole
# run. --signal=B:TERM@300 (below the #SBATCH block) asks SLURM to signal 5 min
# early so on_exit has time to copy.
trap on_exit EXIT TERM INT

cd "$REPO"
echo "=========================================================="
echo "[exp6-$TASK] WM on DEMOS ONLY, encoder frozen"
echo "  base cfg  = $CFG"
echo "  dataset   = $DS_REL"
echo "  warmstart = $FT_CKPT (encoder+projector FROZEN)"
echo "  out       = $OUT_NAME -> $CKPT_PERSIST"
echo "  epochs    = $MAX_EPOCHS   job=${SLURM_JOB_ID:-local}  host=$(hostname)"
echo "=========================================================="
nvidia-smi | head -5

python train.py --config-name "$CFG" \
    "${EXTRA[@]}" "${SMOKE_ARGS[@]}" \
    trainer.precision=32 \
    trainer.max_epochs="$MAX_EPOCHS" \
    data.dataset.name="$DS_REL" \
    output_model_name="$OUT_NAME" \
    subdir="$SUBDIR" \
    wandb.config.project=lewm-exp6-demoonly \
    wandb.config.name="$OUT_NAME" \
    wandb.config.id="${OUT_NAME}_${SLURM_JOB_ID:-local}" \
    wandb.config.resume=allow
rc=$?
echo "=== train rc=$rc ==="
exit $rc
