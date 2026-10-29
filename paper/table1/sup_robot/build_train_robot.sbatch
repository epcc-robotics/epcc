#!/usr/bin/env bash
# Robot-only SuP scheduler, built exactly like the published full-channel one.
#
# The published scheduler (iqlak_full_park_paper.pt) came from two jobs:
#   11678861  build_iql_dataset_ak.py --channel full --demo <park demo> --clip <c> --tag _park
#   11683986  train_iql_ak.py --omega 5 --gamma 0.9 --tau 0.95 --k_min 2 --k_max 4
# This job repeats both with ONE change, --channel robot, which makes the builder
# load adm_A.pt (robot-only ADM) and drop the block term from E. Demo, clip,
# stride (=C=15), max_ep, eps quantile (0.4), policy_q (min) and the HP are the
# same. Outputs are new file names; nothing existing is overwritten.
#
#SBATCH --job-name=sup_robot_build
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
unset LD_PRELOAD
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPO=$REPO_ROOT
PY=python
H=$STABLEWM_HOME
cd "$REPO" || exit 1

declare -A SUPDIR=([pusht]="$H/pusht/sup" [slippery]="$H/slippery_pusht/sup" [cube]="$H/cube/sup")
# identical to run_ak_pilot.sbatch (job 11678861), which built the published D'
declare -A DEMO=(
  [pusht]="$H/datasets/expert_slow_025_n200_centered_v2_park.h5"
  [slippery]="$H/slippery_pusht/data/expert_slow_050_n200_d035_park.h5"
  [cube]="$H/cube/data/cube_multiview_diffusion_c03.h5"
)
declare -A CLIP=([pusht]="inf" [slippery]="inf" [cube]="1.0")

# refuse to overwrite anything
for task in pusht slippery cube; do
  for f in "${SUPDIR[$task]}/iqlak_robot_park_dataset.npz" "${SUPDIR[$task]}/iqlak_robot_park_paper.pt"; do
    [ -e "$f" ] && { echo "!! exists, refusing to overwrite: $f"; exit 1; }
  done
  [ -f "${SUPDIR[$task]}/adm_A.pt" ] || { echo "!! missing ${SUPDIR[$task]}/adm_A.pt"; exit 1; }
done

fail=0
echo "=== robot-only SuP scheduler  JOB=${SLURM_JOB_ID:-local}  $(date) ==="
for task in pusht slippery cube; do
  d="${SUPDIR[$task]}"
  echo ""
  echo "##################### $task / robot : build D' (A^k conditioned) #####################"
  md5sum "$d/adm_A.pt" "$(readlink -f "${DEMO[$task]}")"
  $PY scripts/sup/build_iql_dataset_ak.py \
      --task "$task" --channel robot --demo "${DEMO[$task]}" \
      --clip "${CLIP[$task]}" --tag _park || { fail=1; continue; }

  echo ""
  echo "########## $task / robot : paper (Omega 5, gamma 0.9, tau 0.95, k in 2..4) ##########"
  $PY scripts/sup/train_iql_ak.py --dataset "$d/iqlak_robot_park_dataset.npz" \
      --omega 5 --gamma 0.9 --tau 0.95 --k_min 2 --k_max 4 \
      --out "$d/iqlak_robot_park_paper.pt" || fail=1
done
echo ""
echo "=== DONE fail=$fail  $(date) ==="
exit $fail
