#!/usr/bin/env python3
"""Build a frame-budget-matched subsample of an episodic HDF5 dataset.

Experiment 6 asks whether broad play data is *necessary* or success demos
suffice. Trained head-to-head as-is, that question is unanswerable: the play
corpus has 7.9x (cube) to 28x (plain pushT) more frames than the demo set, so
"play wins" would be indistinguishable from "more data wins".

This script builds the missing control: play, cut down to exactly the demo
frame budget. Then

    demo-only  vs  play-subsampled   -> the CHARACTER of the data
    play-sub   vs  play-full         -> the VOLUME of the data

which decomposes the confound instead of arguing about it.

Design notes
------------
* Episodes are selected with a seeded RNG, never a prefix. Taking the first N
  frames once produced a degenerate latent cache whose first 60 frames were
  near-clones (memory: feedback_latent_cache_first60_bug), which silently
  flattens every spread-based statistic downstream.
* Whole episodes only — cutting mid-episode would break the (K history + H
  horizon) windowing the WM dataloader does off ep_offset/ep_len.
* Per-frame vs per-episode columns are classified by array length rather than a
  hardcoded list, so this works on the cube 37-key schema, the pushT 8-key
  schema, and the reconciled slippery corpus schema without edits.
* Pixel columns are streamed episode-by-episode; the 172 GB cube play corpus
  never lands in RAM.

Usage:
    python scripts/data/subsample_episodes_h5.py \\
        --src   "$MV/cube_multiview_WM.h5" \\
        --out   "$MV/cube_multiview_WM_sub256k.h5" \\
        --target_frames 256000 --seed 42
"""
import argparse
import time
from pathlib import Path

import hdf5plugin  # noqa: F401  — registers Blosc/LZ4 for compressed sources
import h5py
import numpy as np


def classify(f, n_frames, n_eps):
    """Split datasets into per-frame / per-episode / other by leading length."""
    per_frame, per_ep, other = [], [], []
    for k in f:
        obj = f[k]
        if not isinstance(obj, h5py.Dataset):
            continue
        n = obj.shape[0] if obj.shape else 0
        if n == n_frames:
            per_frame.append(k)
        elif n == n_eps:
            per_ep.append(k)
        else:
            other.append(k)
    # Ambiguous when n_frames == n_eps; refuse rather than guess.
    if n_frames == n_eps:
        raise SystemExit("n_frames == n_eps: cannot classify columns by length")
    return sorted(per_frame), sorted(per_ep), sorted(other)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_frames", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=512, help="frames per pixel write")
    a = ap.parse_args()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.src, "r") as src:
        ep_len = np.asarray(src["ep_len"][:], np.int64)
        ep_off = np.asarray(src["ep_offset"][:], np.int64)
        n_eps = int(ep_len.size)
        n_frames = int(src["pixels"].shape[0]) if "pixels" in src else int(ep_len.sum())

        per_frame, per_ep, other = classify(src, n_frames, n_eps)
        print(f"source     : {a.src}")
        print(f"  frames={n_frames:,}  episodes={n_eps:,}")
        print(f"  per-frame : {per_frame}")
        print(f"  per-episode: {per_ep}")
        if other:
            print(f"  UNCLASSIFIED (copied verbatim): {other}")

        # ---- seeded episode selection up to the frame budget ----
        rng = np.random.default_rng(a.seed)
        order = rng.permutation(n_eps)
        take, total = [], 0
        for e in order:
            L = int(ep_len[e])
            if total + L > a.target_frames and take:
                continue          # skip, keep scanning for one that still fits
            take.append(int(e))
            total += L
            if total >= a.target_frames:
                break
        take = np.array(sorted(take), dtype=np.int64)   # sorted -> sequential reads
        out_frames = int(ep_len[take].sum())
        print(f"\nselected   : {len(take):,} episodes, {out_frames:,} frames "
              f"(target {a.target_frames:,}, seed {a.seed})")

        with h5py.File(a.out, "w") as dst:
            # per-episode columns
            for k in per_ep:
                if k == "ep_offset":
                    continue      # rebuilt below
                dst.create_dataset(k, data=np.asarray(src[k])[take])
            new_off = np.zeros(len(take), np.int64)
            np.cumsum(ep_len[take][:-1], out=new_off[1:])
            dst.create_dataset("ep_offset", data=new_off)

            for k in other:
                dst.create_dataset(k, data=np.asarray(src[k]))

            # per-frame columns: allocate, then stream episode by episode
            outs = {}
            for k in per_frame:
                d = src[k]
                chunks = None
                if d.chunks:
                    chunks = (min(d.chunks[0], out_frames),) + d.chunks[1:]
                # Compression must be carried over, not guessed. These files are
                # written with an hdf5plugin filter (Blosc), and h5py reports
                # `d.compression is None` for any plugin filter it doesn't know
                # natively — so the obvious `"lzf" if d.compression` test writes
                # the output UNCOMPRESSED. That is how the first cube control
                # file came out 72 GB instead of ~22 GB (raw 301 KB/frame for
                # pixels+wrist_pixels), which then has to be staged to node-local
                # /tmp by the training launcher.
                copts = {}
                if d.chunks:
                    if d.compression:                      # gzip / lzf / szip
                        copts["compression"] = d.compression
                        if d.compression_opts is not None:
                            copts["compression_opts"] = d.compression_opts
                    elif getattr(d, "_filters", None):     # plugin filter (Blosc, zstd, ...)
                        copts.update(hdf5plugin.Blosc(cname="lz4", clevel=5,
                                                      shuffle=hdf5plugin.Blosc.BITSHUFFLE))
                outs[k] = dst.create_dataset(
                    k, shape=(out_frames,) + d.shape[1:], dtype=d.dtype,
                    chunks=chunks, **copts,
                )
                if k in ("pixels", "wrist_pixels"):
                    print(f"  [alloc] {k}: chunks={chunks} src_filters={getattr(d,'_filters',None)} "
                          f"-> {'plugin-blosc' if 'compression' not in copts and copts else copts or 'NONE'}")

            t0 = time.time()
            w = 0
            for i, e in enumerate(take):
                s, L = int(ep_off[e]), int(ep_len[e])
                for k in per_frame:
                    d = src[k]
                    if d.ndim >= 3:                     # pixel-like: stream
                        for b in range(0, L, a.chunk):
                            n = min(a.chunk, L - b)
                            outs[k][w + b: w + b + n] = d[s + b: s + b + n]
                    else:
                        outs[k][w: w + L] = d[s: s + L]
                w += L
                if (i + 1) % 50 == 0 or i + 1 == len(take):
                    el = time.time() - t0
                    print(f"  [{i+1:>5}/{len(take)}] {w:,}/{out_frames:,} frames  "
                          f"{el:6.1f}s  eta {el/(i+1)*(len(take)-i-1):6.1f}s", flush=True)
            assert w == out_frames, f"wrote {w} != {out_frames}"

            dst.attrs["source"] = a.src
            dst.attrs["subsample_seed"] = a.seed
            dst.attrs["target_frames"] = a.target_frames
            dst.attrs["n_episodes"] = len(take)
            dst.attrs["selected_episode_ids"] = take

    print(f"\nwrote {a.out}  ({out_frames:,} frames, {len(take):,} episodes)")


if __name__ == "__main__":
    main()
