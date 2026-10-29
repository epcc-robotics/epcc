#!/usr/bin/env bash
# Per-corpus action scaler statistics for the friction sweep.
#
# `action_clamp` lives in the CEM's NORMALIZED action space: the scaler is a
# StandardScaler fitted on `dataset.stats`, so the env limit |a_i| <= 1 maps to
# |z_i| <= 1/std_i. Swapping the WM without moving the clamp silently shrinks the
# reachable physical range (a documented case cut it to 40% of the champion's).
#
# Two numbers come out of this, both needed before any Arm-B cell can be written:
#   action_clamp  KEEP the champion's literal value if std moves <1%, else
#                 C_new = C_champ * std_champ / std_new
#   cref          solver.reference.c is "expert pace / WM-training pace", so a
#                 friction corpus whose action scale matches the champion's needs
#                 no re-match. The ratio is printed so that stays a measurement
#                 rather than an assumption.
#
#SBATCH --job-name=fric_clamp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; source "$REPO_ROOT/scripts/env.sh" || exit 1
H=$STABLEWM_HOME

python -u - <<PY
import numpy as np, hdf5plugin, h5py  # noqa: F401
from sklearn import preprocessing
H = "$H"
D = f"{H}/slippery_pusht/data"
CHAMP_CLAMP = np.array([5.0528, 5.2017])      # slippery_pusht_dtg_mpcc.yaml:79
CASES = [("CHAMPION corpus v1", f"{D}/slippery_pusht_wm_corpus.h5")] + \\
        [(f"friction d{d:03d}", f"{D}/friction_d{d:03d}_excised.h5")
         for d in (0, 10, 20, 30, 40, 50, 60, 70, 80)]

rows = []
for tag, path in CASES:
    with h5py.File(path, "r") as f:
        a = np.asarray(f["action"][:], np.float64)
    if a.ndim == 1:
        a = a[:, None]
    a = a[~np.isnan(a).any(1)]
    sc = preprocessing.StandardScaler().fit(a)
    rows.append((tag, len(a), sc.mean_, sc.scale_, np.abs(a).max(0),
                 np.percentile(np.abs(a), 99, axis=0)))

champ_std = rows[0][3]
print(f"{'corpus':20s} {'n':>10s} {'std':>20s} {'clamp @|a|<=1':>20s} "
      f"{'std/champ':>16s} {'RESCALE clamp':>20s} {'KEEP?':>6s}")
for tag, n, mu, std, mx, p99 in rows:
    ratio = std / champ_std
    resc = CHAMP_CLAMP * champ_std / std
    keep = "KEEP" if np.all(np.abs(ratio - 1.0) < 0.01) else "RESC"
    print(f"{tag:20s} {n:10,d} {np.round(std,6).tolist()!s:>20s} "
          f"{np.round(1.0/std,4).tolist()!s:>20s} "
          f"{np.round(ratio,4).tolist()!s:>16s} "
          f"{np.round(resc,4).tolist()!s:>20s} {keep:>6s}")

print("\n--- notes ---")
for tag, n, mu, std, mx, p99 in rows:
    print(f"{tag:20s} mean={np.round(mu,5).tolist()}  "
          f"|a|max={np.round(mx,4).tolist()}  clamp@p99={np.round(p99/std,3).tolist()}")

print("\nREAD:")
print("  KEEP  = std moved < 1% vs the reference WM -> keep action_clamp [5.0528, 5.2017]")
print("  RESC  = otherwise use the 'RESCALE clamp' column")
print("  cref  = std/champ near 1.0 means the same training pace -> keep solver.reference.c=0.55")
print("          (re-matching cref = 0.55 / pace_ratio is the D-solo convention)")
PY
echo "=== DONE rc=$? ==="
