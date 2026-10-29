"""Select the DTG checkpoint by min val_sample_l2 and symlink models/selected_val.pt.

DTG-050 lesson: model_final overfits on small sets (val 4x worse) — always pick
by val_sample_l2. metrics.jsonl rows carry a step + val_sample_l2 (logged every
val_freq steps); checkpoints exist at save_freq multiples + final. We pick the
saved ckpt nearest (<=) the best-val step; fallback model_final.pt when no val
rows exist (val_frac=0 runs).

  python select_dtg_ckpt.py --run_dir <.../pusht_latent_velloss/EXP/42>
"""
import argparse
import json
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--min_step", type=int, default=20000)
    args = ap.parse_args()
    run = Path(args.run_dir)
    models = run / "models"
    assert models.is_dir(), f"no models dir: {models}"

    saved = {}
    for p in models.glob("model_*.pt"):
        m = re.match(r"model_(\d+)\.pt$", p.name)
        if m:
            saved[int(m.group(1))] = p
    final = models / "model_final.pt"

    best_step, best_val = None, float("inf")
    mj = run / "metrics.jsonl"
    if mj.exists():
        for ln in mj.read_text().splitlines():
            try:
                r = json.loads(ln)
            except Exception:
                continue
            v = r.get("val_sample_l2")
            s = r.get("step", r.get("gradient_steps"))
            if v is None or s is None:
                continue
            v = float(v)
            if s is not None and int(s) < args.min_step:
                continue   # early ckpts: val minimum there is spurious (2026-07-10 collapse)
            if v == v and v < best_val:      # skip NaN
                best_val, best_step = v, int(s)

    if best_step is None:
        assert final.exists(), f"no val rows and no model_final in {models}"
        target = final
        print(f"[select] no val_sample_l2 rows -> model_final.pt")
    else:
        cands = [s for s in saved if s <= best_step] or list(saved)
        target = saved[max(cands)] if cands else final
        print(f"[select] best val_sample_l2={best_val:.4f} @step {best_step} "
              f"-> {target.name}")

    link = models / "selected_val.pt"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.name)
    print(f"[select] {link} -> {target.name}")


if __name__ == "__main__":
    main()
