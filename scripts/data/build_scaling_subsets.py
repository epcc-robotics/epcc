#!/usr/bin/env python3
"""Nested, source-stratified episode subsets of an aggregated corpus (data-scaling sweep).

The arXiv scalability experiment trains one world model per subset size of the
aggregated play corpus D (= agg_<task>_main2, O+B+D) and plots EPCC performance
against size. Three properties make the sizes comparable:

* Stratified by SOURCE, not just by O/B/D: a stratum is the per-episode source
  name (`source_id` -> attr `source_names`) without its repeat suffix, e.g.
  O_excised, B_k1..B_k5 (repeats r0, r1, ... pooled), D_c0.5/D_c1.5/D_c2.5.
  Every subset therefore keeps the pace / downsampling-rate mix of the full
  corpus (a pure O/B/D split let the 1/256 subset miss D_c1.5 and B_k2
  entirely), so only the AMOUNT of data changes.
* Nested: one fixed random order per stratum; a subset of size 1/k takes the
  prefix of that order whose frame count is NEAREST to (stratum frames)/k
  (at least one episode). Quotas grow as k shrinks, so smaller subsets are
  contained in larger ones.
* Held-out val: before any subset is drawn, `val_frac` of the episodes of each
  stratum (at least one) are set aside into one validation file that no subset
  contains. Every size is checkpoint-selected on that same file (train.py
  +data.val_dataset_name), and denominator 1 means "all non-val episodes".
* Each written file carries `selection_sha1` (of its sorted episode ids); an
  existing file is reused only if that fingerprint matches the current
  selection. The id lists themselves live in the manifest JSON (an HDF5
  attribute cannot hold more than 64 KiB, i.e. ~8k int64 ids).

The selection is a pure function of (--src, --seed, --val_frac, --denoms), so
several jobs can each write a different part (--write) and still agree on the
split; each job also writes the full manifest.

Usage (CPU node; never on the login node):
    python scripts/data/build_scaling_subsets.py \
        --src  $H/aggcorpus/plain_main2.h5 \
        --out_dir $H/datasets/scaling/plain --prefix plain_main2 \
        --denoms 256 64 16 4 1 --val_frac 0.01 --seed 0 --write val 256 64
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import hdf5plugin  # noqa: F401  (registers Blosc for the compressed sources)
import h5py
import numpy as np


def classify(f, n_frames, n_eps):
    """Split top-level datasets into per-frame / per-episode / other by length."""
    if n_frames == n_eps:
        raise SystemExit("n_frames == n_eps: cannot classify columns by length")
    per_frame, per_ep, other = [], [], []
    for k in f:
        obj = f[k]
        if not isinstance(obj, h5py.Dataset):
            continue
        n = obj.shape[0] if obj.shape else 0
        (per_frame if n == n_frames else per_ep if n == n_eps else other).append(k)
    return sorted(per_frame), sorted(per_ep), sorted(other)


def stratum_of(name):
    """Source name without its repeat suffix: B_k3_r1 -> B_k3; D_c1.5 and O_excised unchanged."""
    return re.sub(r"_r\d+$", "", name)


def fingerprint(ids):
    return hashlib.sha1(np.asarray(sorted(ids), dtype=np.int64).tobytes()).hexdigest()


def select(ep_len, strata, denoms, val_frac, seed, contiguous=(), guard=3):
    """Return (val_ids, {denom: ids}, guard_ids) — nested, stratified per source, deterministic.

    Strata named in `contiguous` take their val episodes as ONE contiguous block in
    corpus order, and drop `guard` episodes on each side of it from every subset.
    Wall excision splits a trajectory into consecutive episodes (excise_walls.py
    re-numbers episode_idx, so the parent id is lost); a contiguous block plus a
    guard band keeps the siblings of val episodes out of training.
    """
    rng = np.random.default_rng(seed)
    val, pools, guarded = [], {}, []
    for s in sorted(set(strata)):                        # sorted -> same rng draws in every job
        ids = np.flatnonzero(strata == s)                # ascending = corpus order
        n_val = max(1, int(round(val_frac * len(ids))))
        if s in contiguous:
            start = int(rng.integers(0, len(ids) - n_val + 1))
            block = ids[start:start + n_val]
            band = np.concatenate([ids[max(0, start - guard):start],
                                   ids[start + n_val:start + n_val + guard]])
            val.extend(block.tolist())
            guarded.extend(band.tolist())
            rest = np.setdiff1d(ids, np.concatenate([block, band]))
            pools[s] = rng.permutation(rest)
        else:
            order = rng.permutation(ids)
            val.extend(order[:n_val].tolist())
            pools[s] = order[n_val:]
    subsets = {}
    for d in sorted(set(denoms), reverse=True):          # smallest subset first
        take = []
        for s, pool in pools.items():
            if d == 1:
                n = len(pool)
            else:
                cum = np.cumsum(ep_len[pool])
                quota = cum[-1] / d
                n = int(np.argmin(np.abs(cum - quota))) + 1   # nearest prefix, >= 1 episode
            take.extend(pool[:n].tolist())
        subsets[d] = np.array(sorted(take), dtype=np.int64)
    return (np.array(sorted(val), dtype=np.int64), subsets,
            np.array(sorted(guarded), dtype=np.int64))


def write_subset(src, out_path, take, per_frame, per_ep, other, ep_len, ep_off,
                 chunk, attrs):
    """Stream the selected episodes of `src` into a new HDF5 file."""
    out_frames = int(ep_len[take].sum())
    tmp = out_path.with_suffix(".h5.partial")
    with h5py.File(tmp, "w") as dst:
        for k in per_ep:
            if k != "ep_offset":
                dst.create_dataset(k, data=np.asarray(src[k])[take])
        new_off = np.zeros(len(take), np.int64)
        np.cumsum(ep_len[take][:-1], out=new_off[1:])
        dst.create_dataset("ep_offset", data=new_off)
        for k in other:
            dst.create_dataset(k, data=np.asarray(src[k]))

        outs = {}
        for k in per_frame:
            d = src[k]
            chunks = (min(d.chunks[0], out_frames),) + d.chunks[1:] if d.chunks else None
            # carry the compression over (h5py reports plugin filters as None).
            # Blosc (id 32001) is copied with the source's own clevel/shuffle/codec,
            # params[4:7]; the source uses byte-shuffle lz4 (11.3:1 on pixels), and a
            # bit-shuffle default came out ~25% larger.
            copts = {}
            if d.chunks:
                flt = getattr(d, "_filters", None) or {}
                blosc = flt.get("32001", flt.get(32001))
                if d.compression:
                    copts["compression"] = d.compression
                    if d.compression_opts is not None:
                        copts["compression_opts"] = d.compression_opts
                elif blosc is not None and len(blosc) >= 7:
                    copts.update(compression=32001,
                                 compression_opts=(0, 0, 0, 0) + tuple(int(x) for x in blosc[4:7]))
                elif flt:
                    copts.update(hdf5plugin.Blosc(cname="lz4", clevel=5,
                                                  shuffle=hdf5plugin.Blosc.SHUFFLE))
            outs[k] = dst.create_dataset(k, shape=(out_frames,) + d.shape[1:],
                                         dtype=d.dtype, chunks=chunks, **copts)

        t0, w = time.time(), 0
        for i, e in enumerate(take):
            s, L = int(ep_off[e]), int(ep_len[e])
            for k in per_frame:
                d = src[k]
                if d.ndim >= 3:                                   # pixels: stream
                    for b in range(0, L, chunk):
                        n = min(chunk, L - b)
                        outs[k][w + b: w + b + n] = d[s + b: s + b + n]
                else:
                    outs[k][w: w + L] = d[s: s + L]
            w += L
            if (i + 1) % 500 == 0 or i + 1 == len(take):
                el = time.time() - t0
                print(f"    [{out_path.name}] {i+1:>6}/{len(take)} eps  {w:,}/{out_frames:,} frames"
                      f"  {el:7.1f}s  eta {el/(i+1)*(len(take)-i-1):7.1f}s", flush=True)
        assert w == out_frames, f"wrote {w} != {out_frames}"
        for k, v in attrs.items():
            dst.attrs[k] = v
    tmp.rename(out_path)                                    # atomic: no half-written .h5
    return out_frames


def action_stats(path):
    """Mean / std (ddof=0, NaN rows dropped) of `action`, as the eval scaler sees it."""
    with h5py.File(path, "r") as f:
        a = np.asarray(f["action"][:], np.float64)
    a = a[~np.isnan(a).any(axis=1)]
    return a.mean(0).tolist(), a.std(0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", required=True, help="e.g. plain_main2")
    ap.add_argument("--denoms", type=int, nargs="+", default=[256, 64, 16, 4, 1])
    ap.add_argument("--val_frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", nargs="+", default=["all"],
                    help="which files this job writes: val and/or denominators, or all")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--val_contiguous", nargs="*", default=[],
                    help="strata whose val is one contiguous block + guard band "
                         "(wall-excised sources, e.g. O_excised for slippery)")
    ap.add_argument("--val_guard", type=int, default=3)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = set(a.write)
    if "all" in want:
        want = {"val"} | {str(d) for d in a.denoms}

    with h5py.File(a.src, "r") as src:
        ep_len = np.asarray(src["ep_len"][:], np.int64)
        ep_off = np.asarray(src["ep_offset"][:], np.int64)
        n_eps = int(ep_len.size)
        n_frames = int(src["pixels"].shape[0])
        names = json.loads(src.attrs["source_names"])
        sid = np.asarray(src["source_id"][:], np.int64)
        group = np.array([names[i][0] for i in sid])         # 'O' / 'B' / 'D'
        strata = np.array([stratum_of(names[i]) for i in sid])
        per_frame, per_ep, other = classify(src, n_frames, n_eps)
        print(f"source {a.src}: {n_eps:,} episodes, {n_frames:,} frames")
        for g in sorted(set(group)):
            m = group == g
            print(f"  group {g}: {m.sum():6,} eps  {ep_len[m].sum():10,} frames "
                  f"({ep_len[m].sum() / ep_len.sum():.1%})  strata "
                  f"{sorted(set(strata[m]))}")
        print(f"  per-frame {per_frame}\n  per-episode {per_ep}\n  other {other}")

        unknown = set(a.val_contiguous) - set(strata)
        if unknown:
            raise SystemExit(f"--val_contiguous names unknown strata: {sorted(unknown)}")
        val_ids, subsets, guard_ids = select(ep_len, strata, a.denoms, a.val_frac, a.seed,
                                             contiguous=set(a.val_contiguous),
                                             guard=a.val_guard)

        # nesting + disjointness checks (cheap, always run)
        vs = set(val_ids.tolist())
        gs = set(guard_ids.tolist())
        ds = sorted(subsets, reverse=True)
        for small, big in zip(ds, ds[1:]):
            assert set(subsets[small]) <= set(subsets[big]), f"1/{small} not inside 1/{big}"
        for d in ds:
            assert not (set(subsets[d]) & vs), f"1/{d} overlaps the val set"
            assert not (set(subsets[d]) & gs), f"1/{d} overlaps the guard band"

        def summary(ids):
            out = {"n_episodes": int(len(ids)), "frames": int(ep_len[ids].sum()),
                   "selection_sha1": fingerprint(ids)}
            for g in sorted(set(group)):
                out[f"frames_{g}"] = int(ep_len[ids][group[ids] == g].sum())
            out["episodes_per_stratum"] = {s: int((strata[ids] == s).sum())
                                           for s in sorted(set(strata))}
            return out

        manifest = {
            "src": a.src, "seed": a.seed, "val_frac": a.val_frac, "denoms": a.denoms,
            "stratification": "source name without repeat suffix (stratum_of)",
            "val_contiguous": sorted(a.val_contiguous), "val_guard": a.val_guard,
            "guard_excluded_ids": guard_ids.tolist(),
            "full": summary(np.arange(n_eps)),
            "val": {**summary(val_ids), "file": f"{a.prefix}_val.h5",
                    "episode_ids": val_ids.tolist()},
            "subsets": {str(d): {**summary(subsets[d]), "file": f"{a.prefix}_f{d}.h5",
                                 "episode_ids": subsets[d].tolist()} for d in ds},
        }
        full_frames = ep_len.sum()
        print("\nselection (seed %d, val_frac %.3f, strata %s):"
              % (a.seed, a.val_frac, sorted(set(strata))))
        print(f"  val      {manifest['val']['n_episodes']:6,} eps {manifest['val']['frames']:10,} frames"
              f"   eps/stratum {list(manifest['val']['episodes_per_stratum'].values())}")
        for d in ds:
            s = manifest["subsets"][str(d)]
            share = {g: s[f'frames_{g}'] / s['frames'] for g in sorted(set(group))}
            print(f"  1/{d:<5}  {s['n_episodes']:6,} eps {s['frames']:10,} frames "
                  f"(x{s['frames'] / full_frames:.5f} of D)  "
                  + "  ".join(f"{g} {v:.1%}" for g, v in share.items())
                  + f"   eps/stratum {list(s['episodes_per_stratum'].values())}")

        jobs = [("val", val_ids)] + [(str(d), subsets[d]) for d in ds]
        for tag, ids in jobs:
            if tag not in want:
                continue
            fname = f"{a.prefix}_val.h5" if tag == "val" else f"{a.prefix}_f{tag}.h5"
            path = out_dir / fname
            sha = fingerprint(ids)
            if path.exists():
                with h5py.File(path, "r") as old:
                    old_sha = old.attrs.get("selection_sha1", None)
                if old_sha != sha:
                    raise SystemExit(f"{path} exists but holds a different selection "
                                     f"({old_sha} != {sha}); delete it to rebuild")
                print(f"\n[skip] {path} exists with the same selection ({sha[:12]})")
            else:
                print(f"\n[write] {path}: {len(ids):,} episodes")
                # ids go to the manifest, not to an attribute (64 KiB attribute cap)
                write_subset(src, path, ids, per_frame, per_ep, other, ep_len, ep_off,
                             a.chunk, {"source": a.src, "seed": a.seed,
                                       "val_frac": a.val_frac, "subset": tag,
                                       "n_episodes": len(ids), "selection_sha1": sha,
                                       "manifest": f"{a.prefix}_manifest.json"})
            mean, std = action_stats(path)
            print(f"  action mean {np.round(mean, 5).tolist()}  std {np.round(std, 5).tolist()}")
            stats_path = out_dir / f"{fname[:-3]}_action_stats.json"
            stats_path.write_text(json.dumps({"file": fname, "action_mean": mean,
                                              "action_std": std}, indent=1))

    # several jobs write the same manifest; write-then-rename keeps it whole
    mpath = out_dir / f"{a.prefix}_manifest.json"
    mtmp = out_dir / f".{a.prefix}_manifest.{'_'.join(sorted(want))}.tmp"
    mtmp.write_text(json.dumps(manifest, indent=1))
    mtmp.rename(mpath)
    print(f"\nmanifest -> {mpath}")


if __name__ == "__main__":
    main()
