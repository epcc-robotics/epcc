"""Precompute the observation/action scalers that evaluation cells fit on large training sets.

Several paper cells normalize observations and actions with statistics of the world-model
training corpus (``task.dataset_stats``), which is far larger than everything else a cell
needs. This fits those scalers once, with the evaluation harness's own fitting function, and
stores them as ``<out>/stats/<stats dataset>.npz``; eval_mpc_multiview.py loads them instead of
opening the corpus (EPCC_NO_STATS_CACHE=1 restores the refit).

    python scripts/data/cache_scaler_stats.py [--source DIR] [--out DIR] [paper/<group>/<cell> ...]

--source holds datasets/<name>.h5 (default $STABLEWM_HOME); --out receives stats/ (default
$STABLEWM_HOME). Default cells: every config under config/eval/cells/paper/.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def paper_cells():
    root = REPO / "config" / "eval" / "cells"
    return sorted(f.relative_to(root).with_suffix("").as_posix() for f in (root / "paper").glob("*/*.yaml"))


def compose(cell):
    from hydra import compose as _compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "config" / "eval" / "cells"), version_base=None):
        return _compose(config_name=cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.environ.get("STABLEWM_HOME"))
    ap.add_argument("--out", default=os.environ.get("STABLEWM_HOME"))
    ap.add_argument("cells", nargs="*")
    a = ap.parse_args()
    import policy._runtime  # noqa: F401  registers the legacy config search path
    import stable_worldmodel as swm
    from eval_mpc_multiview import _fit_scaler

    need = {}  # stats dataset -> set of columns
    for cell in a.cells or paper_cells():
        cfg = compose(cell)
        stats = cfg.dataset.get("stats", cfg.eval.dataset_name)
        if not stats or stats == cfg.eval.dataset_name:
            continue
        cols = [c for c in cfg.dataset.keys_to_cache if c not in ("pixels", "wrist_pixels")]
        need.setdefault(stats, set()).update(cols)
    print(f"{len(need)} stats datasets")
    for stats, cols in sorted(need.items()):
        cols = sorted(cols)
        ds = swm.data.HDF5Dataset(stats, keys_to_cache=cols, cache_dir=a.source)
        arrays = {"source": np.array(stats)}
        for c in cols:
            sc = _fit_scaler(ds.get_col_data(c))
            arrays[f"{c}/mean_"] = sc.mean_
            arrays[f"{c}/var_"] = sc.var_
            arrays[f"{c}/scale_"] = sc.scale_
            arrays[f"{c}/n_samples_seen_"] = np.asarray(sc.n_samples_seen_)
            arrays[f"{c}/n_features_in_"] = np.asarray(sc.n_features_in_)
        out = Path(a.out) / "stats" / f"{stats}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **arrays)
        print(f"  {stats}: {cols} -> {out}")


if __name__ == "__main__":
    main()
