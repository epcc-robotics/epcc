#!/usr/bin/env bash
# Friction sweep, step 3 (2026-09-07 redo): train one H84/K6/stride-1 DTG per friction
# value on the PLAIN park demos, in that friction WM's latent space.
#
# Mirrors train_dtg_friction.sbatch (the H16/K2/stride-5 generation) with the three
# changes the s1H84 unification needs:
#   - CD_ROOT = prj/cleandiffuser-h84 (stock chiunet.py asserts a 2^n horizon; H=84 needs
#     the patched clone, exactly as prj/submit_train_cd_latent_s1H84_pusht.sh)
#   - config A_mm_s1H84_lv30_plain (stride 1 / H84 / K6 / save_freq 2000; only h5_path,
#     exp_name and save_path are overridden here, so the per-d generators differ from the
#     Table I plain generator in NOTHING but the latent space)
#   - NO pruning: every 2000-step checkpoint is kept until the eval confirms the val-argmin
#     choice (decision D2 of the campaign plan); select_dtg_ckpt.py still writes
#     models/selected_val.pt so the eval cells have a deterministic path.
#
#   sbatch --export=ALL,D=d000 --job-name=cd_h84_fricd000 \
#          scripts/experiments/train/train_dtg_friction_s1H84.sbatch
#
#SBATCH --job-name=dtg_fric_s1H84
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/cd_s1H84_fric_%x_%j.out
#SBATCH --error=logs/cd_s1H84_fric_%x_%j.err
set -uo pipefail
D=${D:?set D via --export, e.g. D=d000}
unset LD_PRELOAD

PY=python
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPO=$REPO_ROOT
# PATCHED clone — the stock clone's 2^n assert rejects H=84.
LD_ROOT=$REPO_ROOT/latent_modules/diffusion_cd

NVIDIA_LIB_PATHS=""
ENV_SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")  # site-packages of the active env (pip nvidia libs)
for d in "$ENV_SITE"/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
done
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:-}"

H5="$STABLEWM_HOME/slippery_pusht/data/latent/expert_slow_025_n200_centered_v2_park_latents_lewm_fric${D}.h5"
SAVE="$STABLEWM_HOME/checkpoints/friction/${D}/diffusion_cd_s1H84"
EXP="A_mm_s1H84_plaindemo_fric${D}"
RUN="$SAVE/pusht_latent_velloss/$EXP/42"

[ -e "$H5" ] || { echo "MISSING step-2 latent cache: $H5"; exit 1; }
if [ -d "$RUN/models" ] && [ -n "$(ls -A "$RUN/models" 2>/dev/null)" ]; then
    echo "REFUSING: $RUN/models already holds checkpoints — pick a new exp_name or delete deliberately"
    exit 1
fi
mkdir -p "$SAVE"

echo "=========================================================="
echo "[friction step3 s1H84] DTG on PLAIN park demos, friction ${D} latent space"
echo "  h5   = $H5"
echo "  exp  = $EXP"
echo "  save = $SAVE"
echo "  cd   = $CD_ROOT"
echo "  job  = $SLURM_JOB_ID  host=$(hostname)"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$LD_ROOT"
$PY pipelines/train_latent_velloss.py \
    --config-path ../configs/pusht_latent --config-name A_mm_s1H84_lv30_plain \
    ++h5_path="$H5" ++exp_name="$EXP" ++save_path="$SAVE"
rc=$?
echo "=== train rc=$rc ==="
[ $rc -eq 0 ] || exit $rc

# ---- select by val_sample_l2 (>= 20k); keep every checkpoint (campaign decision D2) ----
python "$REPO_ROOT/paper/common/select_dtg_ckpt.py" --run_dir "$RUN" --min_step 20000
sel_rc=$?
src=$(readlink -f "$RUN/models/selected_val.pt" 2>/dev/null)
if [ $sel_rc -ne 0 ] || [ -z "$src" ] || [ ! -e "$src" ]; then
    echo "FAIL: no usable selected_val.pt for $RUN (select rc=$sel_rc) — the eval cell yamls point at it"
    echo "=== DONE D=$D rc=3 ==="
    exit 3
fi
echo "selected: $src"
ls -la "$RUN/models/" | tail -5
echo "=== DONE D=$D rc=$rc ==="
