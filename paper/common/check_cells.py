"""Check that the paper's evaluation cells still compose to the configurations that were run.

Every cell under config/eval/cells/paper/ inherits a base recipe, so an edit to a base file can
silently change a published cell. This composes each paper cell with Hydra (interpolations left
unresolved) and compares its SHA-256 with config/eval/cells/paper/RESOLVED.sha256.

    python paper/common/check_cells.py            # exit 1 on any difference
    python paper/common/check_cells.py --update   # rewrite the manifest after an intended change
Needs only hydra-core and omegaconf (no GPU, no data).
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.global_hydra import GlobalHydra
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "config" / "eval" / "cells"
MANIFEST = CELLS / "paper" / "RESOLVED.sha256"


class _EvalSearchPath(SearchPathPlugin):  # same search path policy/_runtime.py registers
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="legacy-eval", path=f"file://{REPO}/config/eval")


def cell_hashes():
    Plugins.instance().register(_EvalSearchPath)
    os.environ.setdefault("STABLEWM_HOME", "/unused")
    out = {}
    for f in sorted((CELLS / "paper").glob("*/*.yaml")):
        name = f.relative_to(CELLS).with_suffix("").as_posix()
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(CELLS), version_base=None):
            cfg = compose(config_name=name)
        text = OmegaConf.to_yaml(cfg, resolve=False, sort_keys=True)
        out[name] = hashlib.sha256(text.encode()).hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()
    now = cell_hashes()
    if args.update:
        MANIFEST.write_text("".join(f"{h}  {n}\n" for n, h in now.items()))
        print(f"wrote {MANIFEST.relative_to(REPO)} ({len(now)} cells)")
        return 0
    ref = dict(reversed(line.split()) for line in MANIFEST.read_text().splitlines() if line.strip())
    bad = sorted(n for n in set(ref) | set(now) if ref.get(n) != now.get(n))
    for n in bad:
        print(f"CHANGED  {n}" if n in ref and n in now else f"{'MISSING' if n in ref else 'NEW'}  {n}")
    print(f"{len(now) - len([b for b in bad if b in now])}/{len(now)} paper cells match {MANIFEST.relative_to(REPO)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
