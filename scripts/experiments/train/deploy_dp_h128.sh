#!/usr/bin/env bash
# Retrain the three DP backbones at horizon 128 so a 75-env-step chunk becomes
# available (= the DTG contour length: 16 waypoints x stride 5).
#
# Why 128: ChiUNet1d asserts the action horizon is a power of two
# (cleandiffuser/nn_diffusion/chiunet.py:140), and 75 needs the next one up.
# Why it matters: with dp_chunk_len 75 and CEM horizon H=3 (plan = 15 env-actions),
# floor(75/k) >= 15 holds for every k <= 5, so family P becomes padding-free at
# every rate AND the schedule cap H*k <= 15 admits k up to 5 — the range the
# aggregated WM's B set was collected over.
#
# Everything except `horizon` is byte-identical to the runs that produced the
# current checkpoints (gradient_steps 100000, batch 64, lr 1e-4, seed 42).
# Episode-length census (scripts/analysis/_ep_len_census.py) confirms a 128-step
# window is fully real content in 100% of episodes on all three datasets
# (min real length: cube 130, plain 142, slippery 243).
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1

H=$STABLEWM_HOME
LD=$REPO_ROOT/latent_modules/diffusion_cd
export PY=python

sub () {  # name, launcher, dataset, extra hydra args...
  local name=$1 launcher=$2 ds=$3; shift 3
  echo "== $name"
  [[ "${DRYRUN:-0}" == "1" ]] && return 0
  "$REPO_ROOT/scripts/submit.sh" --cpus-per-task=6 --time=47:00:00 \
    --job-name="$name" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",DATASET_PATH="$ds" \
    "$LD/scripts/$launcher" "$@"
}

sub dph128_cube  submit_train_dp_cube_aw.sh \
    "$H/datasets/cube_multiview_diffusion_c0.3_postpad_fullscale.h5" \
    horizon=128

sub dph128_plain submit_train_dp_pusht_sv.sh \
    "$H/datasets/expert_slow_025_n200_centered_v2_park.h5" \
    horizon=128 'exp_name=dp_pusht_sv_image_park_plain_h128_${now:%Y%m%d_%H%M%S}'

sub dph128_slip  submit_train_dp_pusht_sv.sh \
    "$H/slippery_pusht/data/expert_slow_050_n200_d035_park.h5" \
    horizon=128 'exp_name=dp_pusht_sv_image_park_slip_h128_${now:%Y%m%d_%H%M%S}'
