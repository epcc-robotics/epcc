#!/usr/bin/env bash
# Friction sweep, warm-ring cache (2026-09-07): encode the SLIPPERY park demo set — the
# set the friction eval cells draw their 50 initial states from
# (eval.dataset_name = slippery_pusht/data/expert_slow_050_n200_d035) — into one friction
# WM's latent space.
#
# Why a new cache: policy/dtg_mpcc_warm.py seeds the history ring with
# lat[ep_offset[e] + start - 5 : ...] indexed by the EVAL episode id, so the cache must be
# a cache of the eval dataset. The existing fric${D} caches are of the PLAIN demos (the DTG
# training set) and index different episodes. The _park twin is used, as for the Table I
# warm cells: same episodes as expert_slow_050_n200_d035 with a 100-frame tail appended,
# so frames 0..4 of episode e are the eval frames.
#
# Mirrors encode_slippark_with_aggM2scr.sbatch (same SRC, same REF, same verifier).
# --fp32 is not optional: fp16 compute produced NaN latents for these ViT checkpoints.
#
#   sbatch --export=ALL,D=d000 --job-name=enc_d035_fricd000 \
#          scripts/experiments/collect/encode_d035park_friction.sbatch
#
#SBATCH --job-name=enc_d035_fric
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

SRC="$H/slippery_pusht/data/expert_slow_050_n200_d035_park.h5"      # SLIPPERY park demos (eval set + tail)
CKPT="$H/checkpoints/fric${D}_object.ckpt"                           # that friction WM
OUTDIR="$H/slippery_pusht/data/latent"
OUT="$OUTDIR/expert_slow_050_n200_d035_park_latents_lewm_fric${D}.h5"
REF="$OUTDIR/expert_slow_050_n200_d035_park_latents_lewm_base.h5"    # same demos, slippery base encoder

mkdir -p "$OUTDIR"
echo "=== d035-park x friction encode  D=$D  JOB=$SLURM_JOB_ID ==="
echo "src : $SRC"
echo "ckpt: $CKPT -> $(readlink -f "$CKPT" 2>/dev/null)"
echo "out : $OUT"
for f in "$SRC" "$CKPT" "$REF"; do
    [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists — delete deliberately first"; exit 1; }

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
    print(f"[verify] ref (slippery base enc): shape={zb.shape} dtype={zb.dtype}")
    assert za.shape == zb.shape, "FRAME COUNT / DIM MISMATCH -- do not seed from this"

    def eq(x, y):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            return False
        if x.dtype.kind == 'f':
            return bool((np.isnan(x) == np.isnan(y)).all()
                        and (x[~np.isnan(x)] == y[~np.isnan(y)]).all())
        return bool(np.array_equal(x, y))

    # The warm seeder consumes 'latent' and 'ep_offset'; ep_len/ep_success are the
    # index-identity witnesses.
    for k in ('ep_offset', 'ep_len', 'ep_success'):
        ha, hb = k in a, k in b
        if ha and hb:
            same = eq(a[k], b[k])
            print(f"[verify] CONSUMED {k:<12} identical={same}")
            assert same, f"{k} differs -- the caches index differently, stop"
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
    assert sd > 1e-3, "latent space collapsed -- wrong checkpoint?"
    # Finite over the WHOLE array (38 MB) — the warm seed windows sit at every episode start.
    full = np.asarray(za[:], np.float32)
    assert np.isfinite(full).all(), f"NaN/inf in latents ({int((~np.isfinite(full)).sum())} entries) -- fp32 path failed"
    offs = np.asarray(a['ep_offset'][:], np.int64)
    for o in offs:
        assert np.isfinite(full[o:o+5]).all(), f"NaN in a warm-seed window at offset {o}"
    print("[verify] OK -- index-compatible, finite everywhere (incl. all 200 seed windows), non-degenerate")
PY
vrc=$?
echo "=== verify rc=$vrc ==="
[ $vrc -eq 0 ] || exit $vrc

# ---- parity of THIS encoder against the eval-time encoding path: re-encode the first
# frames of the NON-park eval dataset with the same checkpoint and compare frames 0..4 of
# the leading episodes with the park cache (cube precedent 0.0; fp32 both sides here). ----
EVAL_SRC="$H/datasets/slippery_pusht/data/expert_slow_050_n200_d035.h5"
TMP="$OUTDIR/_parity_tmp_d035_nonpark_fric${D}.h5"
rm -f "$TMP"
python -u scripts/cache_latents.py \
    --dataset pusht --source "$EVAL_SRC" --output "$TMP" --ckpt "$CKPT" \
    --pool lewm --fp32 --batch_size 128 --prefetch 6 --limit 2500
prc=$?
echo "=== parity encode rc=$prc ==="
[ $prc -eq 0 ] || exit $prc
python -u - "$TMP" "$OUT" "$D" <<'PY'
import sys, h5py, numpy as np
tmp, park, D = sys.argv[1], sys.argv[2], sys.argv[3]
with h5py.File(tmp) as a, h5py.File(park) as b:
    la, lb = a['latent'], b['latent']
    oa, ob = np.asarray(a['ep_offset'][:], np.int64), np.asarray(b['ep_offset'][:], np.int64)
    n_lim = la.shape[0]
    worst, n_ep = 0.0, 0
    for e in range(len(oa)):
        if oa[e] + 5 > n_lim: break
        x = np.asarray(la[oa[e]:oa[e]+5], np.float64); y = np.asarray(lb[ob[e]:ob[e]+5], np.float64)
        worst = max(worst, float(np.abs(x - y).max())); n_ep += 1
    ctrl = float(np.abs(np.asarray(la[oa[0]:oa[0]+5], np.float64) - np.asarray(lb[ob[1]:ob[1]+5], np.float64)).max())
    print(f"[parity fric{D}] eval-nonpark vs park cache, frames 0..4 over {n_ep} leading episodes: max|diff|={worst:.3e}  (other-episode control {ctrl:.3e})")
    assert n_ep >= 3, "too few episodes covered by --limit"
    assert worst < 5e-3, f"PARITY FAIL: {worst:.3e} — cache is not what the eval encoder produces"
    print(f"[parity fric{D}] OK")
PY
prc2=$?
rm -f "$TMP"
echo "=== parity rc=$prc2 ==="
[ $prc2 -eq 0 ] || exit $prc2

cat > "$OUTDIR/PROVENANCE_expert_slow_050_n200_d035_park_latents_lewm_fric${D}.md" <<EOF
# slippery (d035) park demos -> friction ${D} latent space  (warm-ring seed cache)

Produced by scripts/experiments/collect/encode_d035park_friction.sbatch,
job ${SLURM_JOB_ID}, $(date -Iseconds).

    demos    ${SRC}
    encoder  ${CKPT}
             -> $(readlink -f "$CKPT" 2>/dev/null)
    pool     lewm (projector(CLS)), --fp32 compute, fp16 storage
    output   ${OUT}
    ref      ${REF}  (ep_offset/ep_len identical per the [verify] block above; parity vs the eval encoder checked)

Seeds the first-replan history ring (policy/dtg_mpcc_warm.py) of the s1H84+warm friction
cell fricB_s1warm_${D}. It is NOT a DTG training set: the Arm-B generators train on the
PLAIN park demos (expert_slow_025_n200_centered_v2_park_latents_lewm_fric${D}.h5).
EOF
echo "=== DONE D=$D rc=0 ==="
