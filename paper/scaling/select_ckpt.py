#!/usr/bin/env python3
"""Pick the held-out-val-minimum epoch of a scaling WM run and register it.

Standard library only (safe on the login node). Reads the training job's stdout,
where Lightning prints one table per validation pass: block 0 is the sanity
check (num_sanity_val_steps=1) and block e is the end of epoch e, i.e. the
weights that ModelObjectCallBack dumps as <name>_epoch_<e>_object.ckpt.

    python findings/261001_scaling/scripts/select_ckpt.py \
        --log logs_smoke/scal_plain_f256_<jobid>.out \
        --run_dir $H/scaling_runs/scal_plain_f256_<jobid> --name scal_plain_f256 \
        --group scaling/plain_f256 [--link]

--link creates $STABLEWM_HOME/checkpoints/<group>_object.ckpt -> the selected
file (what eval's task.policy_config_group=<group> resolves to).
"""
import argparse
import json
import os
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def val_curve(log_path, key="validate/pred_loss_epoch"):
    vals = []
    for line in Path(log_path).read_text(errors="replace").splitlines():
        line = ANSI.sub("", line)
        if key in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) >= 2 and cells[0] == key:
                vals.append(float(cells[1]))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--name", required=True, help="output_model_name of the run")
    ap.add_argument("--group", required=True, help="policy_config_group to register")
    ap.add_argument("--link", action="store_true")
    a = ap.parse_args()

    vals = val_curve(a.log)
    if len(vals) < 2:
        raise SystemExit(f"only {len(vals)} validation blocks in {a.log}")
    sanity, per_epoch = vals[0], vals[1:]              # per_epoch[i] -> epoch i+1
    best_i = min(range(len(per_epoch)), key=per_epoch.__getitem__)
    best_epoch = best_i + 1
    ckpt = Path(a.run_dir) / f"{a.name}_epoch_{best_epoch}_object.ckpt"
    if not ckpt.exists():
        raise SystemExit(f"selected checkpoint missing: {ckpt}")

    rec = {
        "log": str(a.log), "run_dir": str(a.run_dir), "group": a.group,
        "sanity_val": sanity, "n_epochs": len(per_epoch),
        "best_epoch": best_epoch, "best_val": per_epoch[best_i],
        "last_val": per_epoch[-1], "val_curve": per_epoch, "ckpt": str(ckpt),
    }
    print(f"{a.name}: sanity {sanity:.5f} -> best epoch {best_epoch}/{len(per_epoch)} "
          f"val {per_epoch[best_i]:.6f} (last {per_epoch[-1]:.6f})")
    out = Path(a.run_dir) / "selected_ckpt.json"
    out.write_text(json.dumps(rec, indent=1))
    print(f"  -> {out}")

    if a.link:
        home = Path(os.environ["STABLEWM_HOME"])
        link = home / "checkpoints" / f"{a.group}_object.ckpt"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(ckpt)
        print(f"  registered {link} -> {ckpt}")


if __name__ == "__main__":
    main()
