#!/usr/bin/env bash
# EXPERIMENT 6 (cube cell) -- per-view cube WM trained on DEMOS ONLY.
#
# Question: is broad play data necessary, or do success demos suffice?
#   champion WM : Multiview WM/cube_multiview_WM.h5                  (172.7 GB, broad play)
#   this run    : Multiview WM/cube_multiview_diffusion_c0.3_postpad_fullscale.h5  (20.0 GB, demos)
#
# Provenance of the champion (traced 2026-07-31 from
# config/eval/cells/task/cube.yaml -> farm symlinks -> the run's saved
# config.yaml): ckpt `cube_wm/lewm_pv_aw_cube_scratch_epoch_11_object.ckpt`,
# run dir `scratch/stable-wm/pv_aw_cube_scratch_wm_training/`, launched by
# `prj/submit_train_perview_WM_cube_aw_scratch.sh`. The repo config is its
# correct base; the champion's distinguishing settings are Hydra overrides in
# that launcher (`warmstart.ckpt_path=""`, `tokens.proprio.enabled=false`,
# `lr_groups.{encoder,action_encoder}=1.0`) -- reproduced below.
#
# ── what is held fixed vs what is under test ─────────────────────────────────
# FIXED  encoder + projector, warm-started from the champion and FROZEN.
#        Deliberate (plan decision 4): a demo-only encoder would produce a
#        different 192-d space, and every cost weight / nu_scale / action_clamp
#        is tuned in the champion's space. Comparability beats purity here, and
#        Sec V-C says "matching the encoder" explicitly.
# TESTED predictor / pred_proj / action_encoder / grasp head: FROM SCRATCH.
#        They must NOT be warm-started -- inheriting the play-trained predictor
#        would smuggle play dynamics into the "demo-only" model and void the
#        experiment.
#
# ⚠️ lr_groups.{encoder,action_encoder}=1.0 is mandatory whenever a module is
# scratch: the config's 0.1 multipliers assume warm-started modules, and a
# random-init action_encoder at 5e-6 was the documented root cause of failed
# scratch run 6697852 (grad stuck at 0.012).
#
# ⚠️ Expect overfitting. The champion's warm-start sibling (pv_aw_cube_6682735)
# already showed val/pred_loss rising after epoch 1 on 8.6x MORE data. Select
# the ckpt by validation, not by copying the champion's epoch_11.
#
#SBATCH --job-name=pv_cube_demoonly
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:TERM@300
set -uo pipefail

NVIDIA_LIB_PATHS=""
ENV_SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")  # site-packages of the active env (pip nvidia libs)
for d in "$ENV_SITE"/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
done
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:-}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPO=$REPO_ROOT
HOME_WM=$STABLEWM_HOME

# Which dataset this cell trains on. Override DATASET=... to run the matched
# play-subsample control cell without editing this file.
DATASET_DIR="$STABLEWM_HOME/cube/collect"
DATASET="${DATASET:-cube_multiview_diffusion_c0.3_postpad_fullscale}"
SRC_FILE="${DATASET_DIR}/${DATASET}.h5"
WARM_CKPT="${WARM_CKPT:-$HOME_WM/checkpoints/cube_wm/lewm_pv_aw_cube_scratch_epoch_11_object.ckpt}"

# OUT_NAME must track DATASET. CKPT_PERSIST is derived from OUT_NAME, so a
# constant default would make the play-subsample control cell write into the
# demo-only cell's persist dir and overwrite its checkpoints — silently, since
# both are `cp -v ... || true` at exit.
case "$DATASET" in
    cube_multiview_diffusion_c0.3_postpad_fullscale) _CELL=demoonly ;;
    cube_multiview_WM_sub256k)                        _CELL=playsub256k ;;
    cube_multiview_WM)                                _CELL=playfull ;;
    *)                                                _CELL="$DATASET" ;;
esac
MAX_EPOCHS="${MAX_EPOCHS:-100}"

# SMOKE=1 exercises the whole path — config composition, dataset resolution,
# encoder build, warm-start key matching, freeze, optimizer group construction,
# one train step, one val step, one object dump — in minutes instead of hours.
# Three consecutive full-size submissions died on port-environment differences
# (utils/ package shadowing, the datasets/ path segment, spt.vit_hf); each cost
# an A100 allocation to learn one line. Run SMOKE first, always.
# Smoke output is namespaced so it can never land on a real cell's checkpoints.
SMOKE_ARGS=()
if [ "${SMOKE:-0}" = "1" ]; then
    MAX_EPOCHS=1
    SMOKE_ARGS=(+trainer.limit_train_batches=5 +trainer.limit_val_batches=2)
    _CELL="_smoke_${_CELL}"
    echo "[SMOKE] 1 epoch / 5 train batches / 2 val batches"
fi

OUT_NAME="${OUT_NAME:-lewm_pv_aw_cube_${_CELL}}"
SUBDIR="${OUT_NAME}_wm"
CKPT_PERSIST="${CKPT_PERSIST:-$STABLEWM_HOME/train/wm/${OUT_NAME}}"

for f in "$SRC_FILE" "$WARM_CKPT"; do
    [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done

# ── fail fast on schema, BEFORE staging 20 GB ────────────────────────────────
# The demo set was collected for the DIFFUSION pipeline; the per-view WM needs
# wrist pixels, the proprio columns, and the grasp-head label inputs. If any are
# absent we must decide (drop the aux head? regenerate labels?) rather than
# discover it after an hour of staging.
python -u - "$SRC_FILE" <<'PY' || exit 1
import sys, hdf5plugin, h5py  # noqa: F401
need = ["pixels", "wrist_pixels", "action", "observation",
        "proprio_gripper_opening", "proprio_effector_pos",
        "proprio_gripper_contact", "privileged_block_0_pos"]
with h5py.File(sys.argv[1], "r") as f:
    keys = set(f.keys())
    print(f"[schema] {sys.argv[1]}")
    print(f"[schema] frames={f['pixels'].shape[0]:,}" if "pixels" in f else "[schema] no pixels")
    if "ep_len" in f:
        print(f"[schema] episodes={f['ep_len'].shape[0]:,}")
    missing = [k for k in need if k not in keys]
    for k in need:
        print(f"    {'OK  ' if k in keys else 'MISS'}  {k}")
    if missing:
        print(f"\n[schema] MISSING {missing}")
        print("[schema] -> per-view WM cannot train as configured. Either point")
        print("   DATASET at a set that has them, disable the grasp aux head")
        print("   (aux_heads.grasp.enabled=false), or regenerate the columns.")
        print("   Do NOT silently drop them: the champion trained WITH the grasp")
        print("   head, so dropping it adds a second changed variable.")
        sys.exit(1)
    print("\n[schema] OK")
PY

# ── stage on node-local /tmp (Lustre random-read bottleneck) ─────────────────
LOCAL_BASE="/tmp/$USER/stable-wm-${SLURM_JOB_ID:-local_$$}"
# le_wm_v2's swm resolves an HDF5Dataset as {cache_dir}/datasets/{name}.h5.
# The original cube launcher (prj/submit_train_perview_WM_cube_aw_scratch.sh)
# stages to {cache_dir}/{name}.h5 — that repo's swm had no `datasets/` segment.
# Copying that layout verbatim is what broke job 11582828.
LOCAL_DIR="${LOCAL_BASE}/datasets/Multiview WM"
mkdir -p "$LOCAL_DIR"
echo "[staging] $(du -h "$SRC_FILE" | cut -f1) -> $LOCAL_DIR"
cp "$SRC_FILE" "$LOCAL_DIR/" || exit 1

# STABLEWM_HOME stays on PERSISTENT storage; only the DATASET is redirected to
# /tmp (via +data.dataset.cache_dir below). This mirrors submit_finetune_v2.sbatch.
# The original cube launcher exports STABLEWM_HOME=$LOCAL_BASE instead, which
# sends BOTH the dataset and `run_dir = get_cache_dir()/subdir` — i.e. every
# per-epoch checkpoint — to node-local /tmp, where they survive only if the exit
# trap fires. perview_cube.yaml declares no cache_dir key, so the override needs
# the `+` form.
export STABLEWM_HOME="$HOME_WM"

on_exit() {
    mkdir -p "$CKPT_PERSIST"
    # Checkpoints already live on persistent storage ($HOME_WM/$SUBDIR); this
    # only mirrors them into a tidier per-cell directory. Nothing is lost if it
    # does not run.
    cp -v "$HOME_WM/$SUBDIR"/${OUT_NAME}* "$CKPT_PERSIST/" 2>/dev/null || true
    cp -v "$HOME_WM/$SUBDIR/config.yaml" "$CKPT_PERSIST/" 2>/dev/null || true
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
echo "[exp6-cube] per-view cube WM, DEMOS ONLY"
echo "  dataset   = Multiview WM/${DATASET}"
echo "  warmstart = $WARM_CKPT  (encoder+projector only, then FROZEN)"
echo "  out       = $OUT_NAME -> $CKPT_PERSIST"
echo "  job       = ${SLURM_JOB_ID:-local}  host=$(hostname)"
echo "=========================================================="
nvidia-smi | head -5

python train_perview_WM_cube.py \
    "${SMOKE_ARGS[@]}" \
    subdir="$SUBDIR" \
    output_model_name="$OUT_NAME" \
    data.dataset.name="Multiview WM/${DATASET}" \
    +data.dataset.cache_dir="$LOCAL_BASE" \
    trainer.precision=32 \
    trainer.max_epochs="$MAX_EPOCHS" \
    tokens.proprio.enabled=false \
    warmstart.ckpt_path="$WARM_CKPT" \
    +freeze_encoder=true \
    warmstart.modules.encoder=true \
    warmstart.modules.projector=true \
    warmstart.modules.action_encoder=false \
    warmstart.modules.pred_proj=false \
    warmstart.modules.predictor=false \
    lr_groups.encoder=1.0 \
    lr_groups.action_encoder=1.0 \
    wandb.config.project=lewm-perview-cube \
    wandb.config.name="$OUT_NAME" \
    wandb.config.id="${OUT_NAME}_${SLURM_JOB_ID:-local}" \
    wandb.config.resume=allow
rc=$?
echo "=== train rc=$rc ==="
exit $rc
