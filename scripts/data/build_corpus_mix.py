"""N-source WM corpus mixer for the aggregation experiments (O + B + D...).

Generalizes build_corpus_v2.py (hard-wired --v1/--noise) to any number of
heterogeneous sources, adding per-episode provenance columns so a single
Main2 file can be re-sliced into its constituents:

  speed_c   f32  per-episode collection pace (c)
  source_id i16  index into the file attr 'source_names' (JSON list)
  human_demo bool, damping f32  (existing v1 conventions, filled per source)
  ep_success bool  (kept where a source provides it, else False)

Sources are declared in a JSON manifest (attrs-archived verbatim):
  [
    {"path": ".../datasets/slippery_play_c1.0",  # file OR dir of shard_*.h5
     "c": 1.0, "name": "O_play", "human_demo": false, "damping": 0.35},
    {"path": ".../bset/slippery/slippery_bset_k1_r0.h5", "c": 1.0,
     "name": "B_k1"},
    ...
  ]

Per-frame keys are taken from --keys (explicit contract = the WM dataloader's
requirement; default is the pushT set). Every source must provide all of them
(fail fast — no silent zero-fill for load-bearing keys, unlike v2's warn path).
Pixels (and any uint8 image key) are streamed chunk-wise with Blosc-lz4, never
held in RAM. ep_len/ep_offset/episode_idx/step_idx are rebuilt globally.

Usage:
    python scripts/data/build_corpus_mix.py \\
        --manifest mix_manifest.json --out .../slippery_main2.h5 \\
        [--keys pixels action proprio state] [--keys-cube]
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
import hdf5plugin  # noqa: F401
import h5py

PUSHT_KEYS = ["pixels", "action", "proprio", "state"]
CUBE_KEYS = [
    "pixels", "wrist_pixels", "action", "observation",
    "proprio_gripper_opening", "proprio_effector_pos",
    "proprio_gripper_contact", "privileged_block_0_pos",
]
_BLOSC = dict(hdf5plugin.Blosc(cname="lz4", clevel=5,
                               shuffle=hdf5plugin.Blosc.SHUFFLE))
CHUNK = 256  # frames per streamed copy slice


def _source_files(path):
    p = Path(path)
    if not p.is_absolute():                      # manifests store paths relative to $STABLEWM_HOME
        p = Path(os.environ["STABLEWM_HOME"]) / p
    if p.is_dir():
        files = sorted(p.glob("shard_*.h5"))
        if not files:
            raise FileNotFoundError(f"no shard_*.h5 under {p}")
        return files
    if not p.exists():
        raise FileNotFoundError(str(p))
    return [p]


def _is_image_key(f, key):
    d = f[key]
    return d.dtype == np.uint8 and d.ndim == 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", nargs="+", default=None,
                    help="per-frame keys to merge (default: pushT set)")
    ap.add_argument("--keys-cube", action="store_true",
                    help="use the cube per-view key set")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    keys = args.keys or (CUBE_KEYS if args.keys_cube else PUSHT_KEYS)
    manifest = json.loads(Path(args.manifest).read_text())
    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"ERROR: {out_path} exists (use --overwrite)")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: inventory (episode counts, frame counts, shapes, dtypes) ----
    plan = []  # (src_idx, file, ep_len, ep_offset, has_success, success, has_term)
    key_shapes, key_dtypes = {}, {}
    for si, src in enumerate(manifest):
        for fp in _source_files(src["path"]):
            with h5py.File(fp, "r", swmr=True) as f:
                el = f["ep_len"][:].astype(np.int64)
                eo = f["ep_offset"][:].astype(np.int64)
                keep = np.arange(len(el))
                if src.get("exclude_human_demo") and "human_demo" in f:
                    hd = np.asarray(f["human_demo"][:], bool)
                    keep = keep[~hd[keep]]
                if src.get("success_only"):
                    if "ep_success" not in f:
                        raise KeyError(f"{fp.name}: success_only requested but "
                                       "no ep_success column")
                    sc = np.asarray(f["ep_success"][:], bool)
                    keep = keep[sc[keep]]
                el, eo = el[keep], eo[keep]   # ORIGINAL offsets of kept episodes
                n_rows = int(el.sum())
                for k in keys:
                    if k not in f:
                        raise KeyError(f"source '{src.get('name', si)}' file "
                                       f"{fp.name} missing per-frame key '{k}'")
                    d = f[k]
                    if d.shape[0] < eo[-1] + el[-1]:
                        raise ValueError(f"{fp.name}:{k} rows {d.shape[0]} < "
                                         f"episodes span {eo[-1] + el[-1]}")
                    sh, dt = d.shape[1:], d.dtype
                    if k in key_shapes:
                        if key_shapes[k] != sh:
                            raise ValueError(
                                f"key '{k}' shape mismatch: {fp.name} has {sh}, "
                                f"expected {key_shapes[k]}")
                        if key_dtypes[k] != dt:
                            # float<->float: h5py converts on write (first
                            # source's dtype wins). Anything else is an error.
                            if not (np.issubdtype(dt, np.floating)
                                    and np.issubdtype(key_dtypes[k], np.floating)):
                                raise ValueError(
                                    f"key '{k}' dtype mismatch: {fp.name} has "
                                    f"{dt}, expected {key_dtypes[k]}")
                            print(f"[warn] '{k}': {fp.name} {dt} -> "
                                  f"{key_dtypes[k]} (cast on write)")
                    else:
                        key_shapes[k], key_dtypes[k] = sh, dt
                succ = (f["ep_success"][:].astype(bool)[keep]
                        if "ep_success" in f else None)
                has_term = "terminated" in f
                plan.append((si, fp, el, eo, succ, has_term))
                print(f"[scan] {src.get('name', si)} :: {fp.name}: "
                      f"{len(el)} eps / {n_rows} frames"
                      + (" (+ep_success)" if succ is not None else ""))

    all_len = np.concatenate([el for _, _, el, _, _, _ in plan])
    total_eps = len(all_len)
    total_rows = int(all_len.sum())
    new_off = np.zeros(total_eps, np.int64)
    new_off[1:] = np.cumsum(all_len[:-1])
    print(f"[plan] total {total_eps} eps / {total_rows} frames from "
          f"{len(manifest)} sources ({len(plan)} files)")

    # ---- pass 2: write ----
    with h5py.File(out_path, "w") as out:
        out.create_dataset("ep_len", data=all_len.astype(np.int32))
        out.create_dataset("ep_offset", data=new_off)
        out.create_dataset("episode_idx",
                           data=np.repeat(np.arange(total_eps), all_len))
        out.create_dataset("step_idx",
                           data=np.concatenate([np.arange(l) for l in all_len]))

        dsts = {}
        for k in keys:
            sh, dt = key_shapes[k], key_dtypes[k]
            if dt == np.uint8 and len(sh) == 3:
                dsts[k] = out.create_dataset(
                    k, shape=(total_rows, *sh), dtype=dt,
                    chunks=(16, *sh), **_BLOSC)
            else:
                dsts[k] = out.create_dataset(k, shape=(total_rows, *sh), dtype=dt)

        # per-episode provenance
        ep_c = np.empty(total_eps, np.float32)
        ep_srcid = np.empty(total_eps, np.int16)
        ep_hd = np.zeros(total_eps, bool)
        ep_damp = np.full(total_eps, np.nan, np.float32)
        ep_succ = np.zeros(total_eps, bool)
        term_col = np.zeros(total_rows, bool)

        row_ptr = 0
        ep_ptr = 0
        for si, fp, el, eo, succ, has_term in plan:
            src = manifest[si]
            n_eps, n_rows = len(el), int(el.sum())
            with h5py.File(fp, "r", swmr=True) as f:
                # per-episode copy using ORIGINAL offsets (supports arbitrary
                # episode subsets from exclude_human_demo / success_only)
                for k in keys:
                    d_in, d_out = f[k], dsts[k]
                    w = row_ptr
                    for ei in range(n_eps):
                        s0, L = int(eo[ei]), int(el[ei])
                        for s in range(0, L, CHUNK):
                            e = min(s + CHUNK, L)
                            d_out[w + s:w + e] = d_in[s0 + s:s0 + e]
                        w += L
                if has_term:
                    w = row_ptr
                    for ei in range(n_eps):
                        s0, L = int(eo[ei]), int(el[ei])
                        term_col[w:w + L] = np.asarray(
                            f["terminated"][s0:s0 + L], bool)
                        w += L
            sl = slice(ep_ptr, ep_ptr + n_eps)
            ep_c[sl] = float(src.get("c", 1.0))
            ep_srcid[sl] = si
            ep_hd[sl] = bool(src.get("human_demo", False))
            if src.get("damping") is not None:
                ep_damp[sl] = float(src["damping"])
            if succ is not None:
                ep_succ[sl] = succ
            row_ptr += n_rows
            ep_ptr += n_eps
            print(f"[copy] {src.get('name', si)} :: {fp.name} done "
                  f"({row_ptr}/{total_rows} rows)")

        assert row_ptr == total_rows and ep_ptr == total_eps
        out.create_dataset("speed_c", data=ep_c)
        out.create_dataset("source_id", data=ep_srcid)
        out.create_dataset("human_demo", data=ep_hd)
        out.create_dataset("damping", data=ep_damp)
        out.create_dataset("ep_success", data=ep_succ)
        out.create_dataset("terminated", data=term_col)
        out.attrs["source_names"] = json.dumps(
            [s.get("name", str(i)) for i, s in enumerate(manifest)])
        out.attrs["manifest"] = json.dumps(manifest)
        out.attrs["builder"] = "build_corpus_mix.py"

    # census
    print(f"\n[done] {out_path}")
    for i, s in enumerate(manifest):
        m = ep_srcid == i
        print(f"  src {i} {s.get('name', i):>12}: {int(m.sum())} eps / "
              f"{int(all_len[m].sum())} frames, c={s.get('c', 1.0)}, "
              f"success {int(ep_succ[m].sum())}")


if __name__ == "__main__":
    main()
