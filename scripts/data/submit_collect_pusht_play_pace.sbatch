#!/usr/bin/env bash
# PACE launcher for collector #1 (collect_dp_slippery_pusht_play.py) — the v1
# play maker, now with the afterclip pace knob (slowdown_factor, applied after
# the [-1,1] clip). Used for the D-grid play collections (plain + slippery).
#
# Env setup mirrors submit_smoke_collect_slipdp.sbatch (cluster-verified stack:
# pusht_env python + torch 2.10 + dill + diffusion_policy_repro clone).
#
# Usage (all via --export, comma-free values only — sbatch splits on commas):
#   sbatch --export=ALL,C=1.0,DAMPING=0.35,NUM_TRAJ=8,NUM_SHARDS=2 \
#          scripts/data/submit_collect_pusht_play_pace.sbatch          # smoke
#   sbatch --export=ALL,C=1.5,DAMPING=0.35,NUM_TRAJ=400,NUM_SHARDS=4,SEED=43 ...
#   DAMPING=none  -> plain PushT physics (hydra damping=null)
#
# Backbone default = plain paper DP (the v1 play manufacturer), PACE copy via
# the '='-free symlink (hydra rejects '=' inside override values).
#SBATCH --job-name=collect_pusht_play
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1

# collection ran in its own environment (see docs/DATA.md, "Data collection"); use VENV in cluster.env
cd "$REPO_ROOT"

C=${C:-1.0}
PACE_MODE=${PACE_MODE:-displacement}  # displacement | stride (see config comment)
DAMPING=${DAMPING:-0.35}            # "none" -> plain physics
NUM_TRAJ=${NUM_TRAJ:-8}
NUM_SHARDS=${NUM_SHARDS:-2}
SHARD_OFFSET=${SHARD_OFFSET:-0}
SEED=${SEED:-43}                    # v1 used 42 — new collections start at 43
NUM_ENVS=${NUM_ENVS:-25}
MAX_STEPS=${MAX_STEPS:-300}
CKPT=${CKPT:-$STABLEWM_HOME/pusht/policy/dp/plaindp_e0100_s0902.ckpt}

if [ "$DAMPING" = "none" ]; then
  TASKTAG=plain; DAMP_OVR="damping=null"; OUTROOT=$STABLEWM_HOME/pusht
else
  TASKTAG=slippery; DAMP_OVR="damping=$DAMPING"; OUTROOT=$STABLEWM_HOME/slippery_pusht
fi
DATASET_NAME=${DATASET_NAME:-${TASKTAG}_play_c${C}}
mkdir -p "$OUTROOT/datasets"

echo "== collect pusht play  JOB=$SLURM_JOB_ID  task=$TASKTAG c=$C n=$NUM_TRAJ shards=$NUM_SHARDS seed=$SEED ckpt=$CKPT =="
python -c "import stable_worldmodel as s; print('SWM ->', s.__file__)" || exit 1

python scripts/data/collect_dp_slippery_pusht_play.py \
  slowdown_factor="$C" \
  pace_mode="$PACE_MODE" \
  $DAMP_OVR \
  num_traj="$NUM_TRAJ" \
  num_shards="$NUM_SHARDS" \
  shard_offset="$SHARD_OFFSET" \
  world.num_envs="$NUM_ENVS" \
  world.max_episode_steps="$MAX_STEPS" \
  dataset_name="$DATASET_NAME" \
  seed="$SEED" \
  device=cuda:0 \
  dp_ckpt_path="$CKPT" \
  cache_dir="$OUTROOT" \
  hydra.run.dir=logs_dist_shift_collect/hydra/collect_play_${SLURM_JOB_ID}
RC=$?
echo "== collect rc=$RC =="
ls -la "$OUTROOT/datasets/$DATASET_NAME/" 2>/dev/null || echo "no shard dir produced"
exit $RC
