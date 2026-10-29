#!/usr/bin/env bash
# Friction sweep, step 2: re-encode the PLAIN park demos into one friction WM's
# latent space. Each friction WM trains its encoder from scratch, so its latent
# space is new and the champion's DTG cannot be reused — this cache feeds a
# fresh DTG (step 3).
#
# Mirrors encode_plainpark_with_slipenc.sbatch, including its verifier: the same
# demos encoded by a DIFFERENT encoder must still agree on every field the DTG
# windows on, or the Arm-B vs experiment-1 comparison stops being single-variable.
#
# --fp32 is not optional. It sets the COMPUTE dtype (storage is fp16 regardless);
# without it these ViT checkpoints emit NaN latents, which is what produced the
# 0% Ablation-C cell.
#
#   sbatch --export=ALL,D=d000 scripts/experiments/collect/encode_plainpark_friction.sbatch
#
#SBATCH --job-name=enc_fric
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
D=${D:?set D via --export, e.g. D=d000}

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
REPO=$REPO_ROOT
cd "$REPO" || exit 1
H="$STABLEWM_HOME"

SRC="$H/datasets/expert_slow_025_n200_centered_v2_park.h5"     # PLAIN park demos
CKPT="$H/checkpoints/fric${D}_object.ckpt"                      # that friction WM
OUTDIR="$H/slippery_pusht/data/latent"
OUT="$OUTDIR/expert_slow_025_n200_centered_v2_park_latents_lewm_fric${D}.h5"
REF="$OUTDIR/expert_slow_025_n200_centered_v2_park_latents_lewm_slipbase.h5"

mkdir -p "$OUTDIR"
echo "=== friction encode  D=$D  JOB=$SLURM_JOB_ID ==="
echo "src : $SRC"
echo "ckpt: $CKPT -> $(readlink -f "$CKPT" 2>/dev/null)"
echo "out : $OUT"
for f in "$SRC" "$CKPT" "$REF"; do
    [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done

python -u scripts/cache_latents.py \
    --dataset pusht --source "$SRC" --output "$OUT" --ckpt "$CKPT" \
    --pool lewm --fp32 --batch_size 128 --prefetch 6
rc=$?
echo "=== cache_latents rc=$rc ==="
[ $rc -eq 0 ] || exit $rc

python -u - "$OUT" "$REF" "$D" <<'PY'
import sys, h5py, numpy as np
out, ref, D = sys.argv[1], sys.argv[2], sys.argv[3]
with h5py.File(out) as a, h5py.File(ref) as b:
    za, zb = a['latent'], b['latent']
    print(f"\n[verify] new (fric {D} enc): shape={za.shape} dtype={za.dtype}")
    print(f"[verify] ref (slippery enc): shape={zb.shape} dtype={zb.dtype}")
    assert za.shape == zb.shape, "FRAME COUNT / DIM MISMATCH -- do not train on this"

    def eq(x, y):
        # NaN in matching positions counts as equal: array_equal says False when
        # both sides hold NaN in the same slot, which killed an earlier run on a
        # field the DTG never reads.
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            return False
        if x.dtype.kind == 'f':
            return bool((np.isnan(x) == np.isnan(y)).all()
                        and (x[~np.isnan(x)] == y[~np.isnan(y)]).all())
        return bool(np.array_equal(x, y))

    # Strict only on what pusht_latent_dataset.py actually windows on.
    for k in ('ep_offset', 'ep_len', 'ep_success'):
        ha, hb = k in a, k in b
        if ha and hb:
            same = eq(a[k], b[k])
            print(f"[verify] CONSUMED {k:<12} identical={same}")
            assert same, f"{k} differs -- the caches window differently, stop"
        else:
            assert ha == hb, f"{k} present in only one cache (new={ha} ref={hb})"
            print(f"[verify] CONSUMED {k:<12} absent from both (consistent)")
    for k in ('ep_idx', 'episode_idx', 'valid_len', 'action', 'step_idx'):
        if k in a and k in b:
            print(f"[verify] info     {k:<12} identical={eq(a[k], b[k])}")

    x = np.asarray(za[:5000], np.float64)
    nrm, sd = np.linalg.norm(x, axis=1).mean(), x.std(0).mean()
    print(f"\n[verify] latent stats  |z| mean={nrm:.3f}  per-dim std={sd:.4f}  "
          f"nan={int(np.isnan(x).sum())}")
    # A fresh encoder has its own scale, so this is a sanity band rather than a
    # target: NaNs or a collapsed (near-zero std) space mean the wrong ckpt or a
    # failed fp32 path, not a legitimately different latent geometry.
    assert np.isfinite(x).all(), "NaN/inf in latents -- fp32 path failed"
    assert sd > 1e-3, "latent space collapsed -- wrong checkpoint?"
    print("[verify] OK -- index-compatible, finite, non-degenerate")
PY
vrc=$?
echo "=== verify rc=$vrc ==="
[ $vrc -eq 0 ] || exit $vrc

cat > "$OUTDIR/PROVENANCE_expert_slow_025_n200_centered_v2_park_latents_lewm_fric${D}.md" <<EOF
# plain park demos -> friction ${D} latent space

Produced by scripts/experiments/collect/encode_plainpark_friction.sbatch,
job $SLURM_JOB_ID, $(date -Iseconds).

    demos    $SRC
    encoder  $CKPT
             -> $(readlink -f "$CKPT")
    pool     lewm (projector(CLS)), --fp32 compute, fp16 storage
    output   $OUT

Feeds the Arm-B DTG A_mm_park_plaindemo_fric${D}. The reference source is held at
PLAIN demos across the whole sweep; only the encoder (and therefore the latent
space) changes with damping.
EOF
echo "=== DONE D=$D ==="
