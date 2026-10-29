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
    ap.add_argument("--log", required=True, action="append",
                    help="job stdout; repeat in order for a resumed run (each job opens with a sanity table)")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--name", required=True, help="output_model_name of the run")
    ap.add_argument("--group", required=True, help="policy_config_group to register")
    ap.add_argument("--link", action="store_true")
    a = ap.parse_args()

    curves = [val_curve(lg) for lg in a.log]
    if any(len(c) < 2 for c in curves):
        raise SystemExit(f"too few validation blocks in {a.log}: {[len(c) for c in curves]}")
    sanity = curves[0][0]
    if len(curves) == 1:
        per_epoch = curves[0][1:]                      # per_epoch[i] -> epoch i+1
    else:
        # Resumed run: each job opens with a sanity table. The first job's epochs start at 1 and
        # the last job ends at trainer.max_epochs. A resume may RE-RUN the epoch it was saved in
        # (seen 2026-10-22: slippery DEMO re-ran epoch 85 and rewrote its _epoch_85 file), so the
        # last log is aligned to the end and, where the two overlap, the later job's value wins --
        # it is the one whose weights are on disk.
        if len(curves) > 2:
            raise SystemExit("more than two logs: align them by hand")
        cfg = (Path(a.run_dir) / "config.yaml").read_text()
        max_ep = int(re.search(r"^\s*max_epochs:\s*(\d+)", cfg, re.M).group(1))
        first, last = curves[0][1:], curves[1][1:]
        start_last = max_ep - len(last) + 1
        ep = {e: v for e, v in enumerate(first, 1)}
        overlap = sorted(set(ep) & set(range(start_last, max_ep + 1)))
        ep.update({start_last + i: v for i, v in enumerate(last)})
        if sorted(ep) != list(range(1, max_ep + 1)):
            raise SystemExit(f"epochs do not cover 1..{max_ep}: first job {len(first)}, last job "
                             f"{len(last)} starting at {start_last}")
        per_epoch = [ep[e] for e in range(1, max_ep + 1)]
        print(f"first job epochs 1..{len(first)}, last job epochs {start_last}..{max_ep}; "
              f"re-run epochs taken from the last job: {overlap or 'none'}")
    print(f"validation tables per log: {[len(c) for c in curves]} -> {len(per_epoch)} epochs")
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
