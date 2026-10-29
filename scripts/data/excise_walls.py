"""Wall-contact excision for walls-on pushT collections (O/D/plain-D lineage).

Reimplements the original excise_full.py rule, reverse-engineered from
slippery_pusht_wm_corpus_excised.h5 (99.71% frame-level agreement; see
scripts/analysis/excise_rule_forensics.py):

  contact frame := any T-block vertex within the wall surface —
                   min over 8 vertices of box-margin to [5,506] <= 2.0
                   (walls are segments with radius 2; block pose from
                   state[:, 2:5) = body ORIGIN x, y, theta; NOT block_pose,
                   which is all-zero in the corpora — known landmine)
  then per episode: drop contact frames, split the survivors into contiguous
  clean runs, keep runs >= 21 frames, promote each kept run to its own
  episode, dropping the run's LAST row (the action leading into contact /
  the swm NaN terminal row).

Apply ONLY to walls-on collections. Walls-off data (B set) must NOT be
processed — crossing the line is correct physics there.

Usage:
    python scripts/data/excise_walls.py --src IN.h5 --out OUT.h5 [--overwrite]
    python scripts/data/excise_walls.py --src IN.h5 --validate-against REF.h5
"""

import argparse
from pathlib import Path

import numpy as np
import hdf5plugin  # noqa: F401
import h5py

VERTS = np.array([
    [-60.0, 0.0], [60.0, 0.0], [60.0, 30.0], [-60.0, 30.0],
    [-15.0, 30.0], [15.0, 30.0], [-15.0, 120.0], [15.0, 120.0],
])
LO, HI = 5.0, 506.0
CONTACT_T = 2.0     # wall segment radius
MIN_RUN = 21        # runs >= this are promoted (last row then dropped -> >=20)
CHUNK = 256
_BLOSC = dict(hdf5plugin.Blosc(cname="zstd", clevel=5,
                               shuffle=hdf5plugin.Blosc.SHUFFLE))


def vertex_margin(state):
    bx = state[:, 2].astype(np.float64)
    by = state[:, 3].astype(np.float64)
    th = state[:, 4].astype(np.float64)
    ct, sn = np.cos(th), np.sin(th)
    m = np.full(len(state), np.inf)
    for vx, vy in VERTS:
        wx = bx + vx * ct - vy * sn
        wy = by + vx * sn + vy * ct
        m = np.minimum(m, np.minimum.reduce([wx - LO, HI - wx, wy - LO, HI - wy]))
    return m


def clean_runs(margin_ep):
    """Yield (start, end) row ranges (end exclusive, last run row dropped)."""
    L = len(margin_ep)
    ok = margin_ep > CONTACT_T
    ok[L - 1] = False          # terminal row (NaN action / leads out) never survives
    j = 0
    while j < L:
        if not ok[j]:
            j += 1
            continue
        k = j
        while k < L and ok[k]:
            k += 1
        if k - j >= MIN_RUN:
            yield j, k - 1     # drop run-terminal row
        j = k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out")
    ap.add_argument("--validate-against")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    with h5py.File(args.src, "r", swmr=True) as f:
        st = f["state"][:]
        el = f["ep_len"][:].astype(np.int64)
        eo = f["ep_offset"][:].astype(np.int64)
        margins = vertex_margin(st)

        # per-source-episode list of kept (orig_ep, start_row, end_row)
        segs = []
        for i in range(len(el)):
            s, L = int(eo[i]), int(el[i])
            for a, b in clean_runs(margins[s:s + L]):
                segs.append((i, s + a, s + b))
        new_len = np.array([b - a for _, a, b in segs], np.int64)
        total = int(new_len.sum())
        print(f"[excise] {args.src}: {len(el)} eps / {int(el.sum())} fr -> "
              f"{len(segs)} eps / {total} fr "
              f"({total / max(int(el.sum()), 1):.2%} kept)")

        if args.validate_against:
            with h5py.File(args.validate_against, "r", swmr=True) as fr:
                st_r = fr["state"][:]
            def rk(a):
                return np.ascontiguousarray(a).view(
                    np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
            ours = set()
            for _, a, b in segs:
                ours.update(range(a, b))
            k_src, k_ref = rk(st), rk(st_r)
            order = np.argsort(k_src, kind="stable")
            pos = np.clip(np.searchsorted(k_src[order], k_ref), 0, len(k_src) - 1)
            cand = order[pos]
            ref_rows = set(cand[k_src[cand] == k_ref].tolist())
            inter = len(ours & ref_rows)
            print(f"[validate] ours {len(ours)} rows, ref {len(ref_rows)} rows, "
                  f"overlap {inter} (jaccard {inter / len(ours | ref_rows):.4%}; "
                  f"ours-only {len(ours - ref_rows)}, ref-only {len(ref_rows - ours)})")
            return

        out_path = Path(args.out)
        if out_path.exists() and not args.overwrite:
            raise SystemExit(f"ERROR: {out_path} exists (use --overwrite)")

        n_src_eps, n_src_rows = len(el), int(el.sum())
        per_frame = [k for k in f.keys()
                     if f[k].shape[:1] == (n_src_rows,) and k not in
                     ("episode_idx", "step_idx")]
        per_ep = [k for k in f.keys()
                  if f[k].shape[:1] == (n_src_eps,) and k not in
                  ("ep_len", "ep_offset")]
        print(f"[excise] per-frame keys: {per_frame}")
        print(f"[excise] per-episode keys: {per_ep}")

        new_off = np.zeros(len(segs), np.int64)
        new_off[1:] = np.cumsum(new_len[:-1])
        with h5py.File(out_path, "w") as out:
            out.create_dataset("ep_len", data=new_len.astype(np.int32))
            out.create_dataset("ep_offset", data=new_off)
            out.create_dataset(
                "episode_idx", data=np.repeat(np.arange(len(segs)), new_len))
            out.create_dataset(
                "step_idx", data=np.concatenate([np.arange(l) for l in new_len]))
            for k in per_ep:
                col = f[k][:]
                out.create_dataset(
                    k, data=np.array([col[ep] for ep, _, _ in segs]))
            for k in per_frame:
                d_in = f[k]
                sh, dt = d_in.shape[1:], d_in.dtype
                if dt == np.uint8 and len(sh) == 3:
                    d_out = out.create_dataset(
                        k, shape=(total, *sh), dtype=dt, chunks=(16, *sh),
                        **_BLOSC)
                else:
                    d_out = out.create_dataset(k, shape=(total, *sh), dtype=dt)
                w = 0
                for _, a, b in segs:
                    for s2 in range(a, b, CHUNK):
                        e2 = min(s2 + CHUNK, b)
                        d_out[w:w + (e2 - s2)] = d_in[s2:e2]
                        w += e2 - s2
                assert w == total
            out.attrs["rule"] = (
                "wall-contact steps excised (vertex margin <= 2.0 from state); "
                "runs >= 21 promoted; run-terminal row dropped")
            out.attrs["source"] = f"excise_walls.py from {Path(args.src).name}"
            out.attrs["span"] = MIN_RUN - 1
            out.attrs["contact_threshold"] = CONTACT_T
        print(f"[excise] wrote {out_path}")


if __name__ == "__main__":
    main()
