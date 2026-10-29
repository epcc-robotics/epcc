"""Measure each dataset's per-step commanded displacement, so Fig. 4's x axis can
be stated as a multiple of the demonstration pace instead of a collection knob.

The `c` in the file names is the scalar applied at collection time; whether it
survives into the recorded actions is an empirical question (clipping and
at-limit saturation compress it), and the figure axis must report what the data
actually contains, not what was requested.

Reports, per dataset: mean and median per-step |a| (L2 over action dims), the
fraction of steps sitting on the action box, and the ratio of the mean to the
task's demonstration set.
"""
import sys
import os
from pathlib import Path
import numpy as np
import h5py

H = Path(os.environ["STABLEWM_HOME"]) / "datasets"

SETS = {
    "PushT": [
        ("demo  (c=0.25)",        H / "expert_slow_025_n200_centered_v2_park.h5"),
        ("D c=0.5",               H / "agg_plain_D_c0.5.h5"),
        ("D c=1.0",               H / "agg_plain_D_c1.0.h5"),
        ("D c=1.5",               H / "agg_plain_D_c1.5.h5"),
        ("D c=2.5",               H / "agg_plain_D_c2.5.h5"),
        ("O subsample (ctrl pt)", H / "pusht_expert_train_sub83k.h5"),
        ("O full",                H / "pusht_expert_train.h5"),
    ],
    "Cube": [
        ("demo  (c=0.30)",        H / "cube_multiview_diffusion_c0.3_postpad_fullscale.h5"),
        ("D c=0.3",               H / "agg_cube_D_c0.3.h5"),
        ("D c=0.75",              H / "agg_cube_D_c0.75.h5"),
        ("D c=1.2",               H / "agg_cube_D_c1.2.h5"),
        ("O subsample (ctrl pt)", H / "cube_multiview_WM_sub256k.h5"),
        ("O full",                H / "cube_multiview_WM.h5"),
    ],
    "Slippery-PushT": [
        ("demo  (c=0.50)",        H / "slippery_pusht/data/expert_slow_050_n200_d035_park.h5"),
        ("D c=0.5",               H / "agg_slippery_D_c0.5.h5"),
        ("D c=1.5",               H / "agg_slippery_D_c1.5.h5"),
        ("D c=2.5",               H / "agg_slippery_D_c2.5.h5"),
        ("O subsample (ctrl pt)", H / "slippery_pusht/data/slippery_pusht_wm_corpus_v1_sub101k.h5"),
        ("O full",                H / "slippery_pusht/data/slippery_pusht_wm_corpus.h5"),
    ],
}

MAX_STEPS = 200_000   # plenty for a magnitude statistic; keeps the read cheap


def find_actions(f):
    """Return the first dataset named action/actions anywhere in the file."""
    hits = []
    f.visititems(lambda n, o: hits.append(n)
                 if isinstance(o, h5py.Dataset) and n.split("/")[-1] in ("action", "actions")
                 else None)
    return hits


def stats(path: Path):
    if not path.exists():
        return None, f"missing: {path}"
    with h5py.File(path, "r") as f:
        keys = find_actions(f)
        if not keys:
            return None, "no action dataset; top level = " + ",".join(list(f.keys())[:8])
        if len(keys) == 1:                      # flat (N, A)
            a = np.asarray(f[keys[0]][:MAX_STEPS], np.float32)
        else:                                   # per-episode groups
            chunks, n = [], 0
            for k in keys:
                x = np.asarray(f[k][:], np.float32)
                chunks.append(x); n += len(x)
                if n >= MAX_STEPS:
                    break
            a = np.concatenate(chunks, 0)
    if a.ndim == 3:                             # (E, T, A)
        a = a.reshape(-1, a.shape[-1])
    # Every corpus carries NaN rows (the pipeline drops them before fitting the
    # action scaler); a plain mean over them yields NaN, which is what the first
    # pass reported. Drop them here too and report how many there were.
    good = np.isfinite(a).all(-1)
    nan_frac = float(1.0 - good.mean())
    a = a[good]
    if a.size == 0:
        return None, "all rows NaN"
    mag = np.linalg.norm(a, axis=-1)
    at_lim = float((np.abs(a) >= 0.999).any(-1).mean())
    # The demonstrations end with the object held at rest, so a plain mean mixes
    # "how fast it moves" with "how long it stands still" -- and the resting share
    # differs per task (largest on Cube). Report statistics that do not: a high
    # percentile, and the mean restricted to steps that actually command motion,
    # thresholded scale-free at 10% of the set's own 90th percentile.
    p90 = float(np.percentile(mag, 90))
    moving = mag > 0.10 * p90
    return dict(n=len(a), dim=a.shape[-1], mean=float(mag.mean()),
                med=float(np.median(mag)), p75=float(np.percentile(mag, 75)),
                p90=p90, mmean=float(mag[moving].mean()) if moving.any() else float("nan"),
                movefrac=float(moving.mean()), at_lim=at_lim, nan=nan_frac), None


for task, rows in SETS.items():
    print(f"\n########## {task}")
    print(f"{'dataset':24s} {'mean':>8} {'med':>8} {'p90':>8} {'mov.mean':>9} "
          f"{'mov%':>6} {'atlim':>7} || {'xmean':>6} {'xp90':>6} {'xmov':>6}")
    base = None
    for label, path in rows:
        s, err = stats(path)
        if s is None:
            print(f"{label:24s}  -- {err}")
            continue
        if base is None:
            base = s
        print(f"{label:24s} {s['mean']:8.4f} {s['med']:8.4f} {s['p90']:8.4f} "
              f"{s['mmean']:9.4f} {100*s['movefrac']:5.1f}% {100*s['at_lim']:6.1f}% || "
              f"{s['mean']/base['mean']:6.2f} {s['p90']/base['p90']:6.2f} "
              f"{s['mmean']/base['mmean']:6.2f}")
print("\n=== PACE_AUDIT DONE ===")
