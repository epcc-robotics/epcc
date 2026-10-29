#!/usr/bin/env bash
# P-phase per-set driver: excise shards -> single-source mix -> frame-matched
# subsample -> pace_mode attrs. For pushT D sets (checklist P1-P3).
# Usage: sbatch --export=ALL,TASK=slippery,C=1.5,TGT=101310 scripts/data/run_pphase_set.sbatch
#        (TASK=plain -> no damping; cube uses run_pphase_cube instead: no excise)
#SBATCH --job-name=pphase_set
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pphase_%x_%j.out
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPO=$REPO_ROOT; cd "$REPO"
H=$STABLEWM_HOME

TASK=${TASK:?}; C=${C:?}; TGT=${TGT:?}
[ "$TASK" = "slippery" ] && ROOT=$H/slippery_pusht || ROOT=$H/pusht
SRC=$ROOT/datasets/${TASK}_play_stride_c${C}
OUT=$H/dgrid/$TASK; mkdir -p "$OUT/excised_c${C}"

for s in 0 1 2 3 4; do
  python scripts/data/excise_walls.py --src "$SRC/shard_${s}.h5" \
    --out "$OUT/excised_c${C}/shard_${s}.h5" --overwrite || exit 1
done

MAN=$OUT/manifest_c${C}.json
if [ "$TASK" = "slippery" ]; then
  printf '[{"path": "%s", "c": %s, "name": "D_c%s", "damping": 0.35}]\n' "$OUT/excised_c${C}" "$C" "$C" > "$MAN"
else
  printf '[{"path": "%s", "c": %s, "name": "D_c%s"}]\n' "$OUT/excised_c${C}" "$C" "$C" > "$MAN"
fi
python scripts/data/build_corpus_mix.py --manifest "$MAN" \
  --out "$OUT/D_c${C}_excised.h5" --overwrite || exit 1

python scripts/data/subsample_episodes_h5.py --src "$OUT/D_c${C}_excised.h5" \
  --out "$OUT/D_c${C}_final.h5" --target_frames "$TGT" --seed 42 || exit 1

python - "$OUT/D_c${C}_final.h5" "$C" <<'EOF'
import sys, h5py, hdf5plugin  # noqa
with h5py.File(sys.argv[1], "a") as f:
    f.attrs["pace_mode"] = "stride"
    f.attrs["slowdown_factor"] = float(sys.argv[2])
    f.attrs["postprocess"] = "excise_walls + subsample seed42"
    print("final:", sys.argv[1], int(f["ep_len"][:].sum()), "frames /",
          len(f["ep_len"]), "eps; attrs ok")
EOF
echo "=== DONE rc=$? ==="
