"""Per-run gate check (login-node safe: stdlib only). One line per gate, PASS/FAIL, then a verdict.

  python3 gate_check.py <run_dir> <job stdout .out> [--expect-dtg <ckpt path>] [--expect-stats <corpus>] [--warm]
"""
import argparse, json, os, re, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("out")
    ap.add_argument("--expect-dtg", default=None); ap.add_argument("--expect-stats", default=None)
    ap.add_argument("--warm", action="store_true", help="require warm-ring seeding evidence")
    ap.add_argument("--cand", action="store_true", help="candidate policy: also check the [cand] line")
    ap.add_argument("--expect-stride", type=int, default=1, help="generator stride (1 = s1H84 campaign cells; 5 = old H16/K2 generators)")
    a = ap.parse_args()
    S = a.expect_stride
    exp_idx = "stride_indices=[0, 1, 2, 3, 4, 5]" if S == 1 else "stride_indices=[0, 5]" if S == 5 else None
    ok_all = True
    def gate(name, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    out = open(a.out, errors="replace").read()
    err_path = a.out[:-4] + ".err" if a.out.endswith(".out") else None
    err = open(err_path, errors="replace").read() if err_path and os.path.exists(err_path) else ""
    m = re.search(r"=== DONE cell=(\S+) rc=(\d+) ===", out)
    gate("1 DONE rc=0", m is not None and m.group(2) == "0", m.group(0) if m else "no DONE line (TIMEOUT/OOM?)")
    gate("1b no Traceback in .err", "Traceback" not in err, "Traceback present" if "Traceback" in err else "")
    s = re.search(r"\[scaler\] fitting on stats dataset: (\S+)", out)
    if a.expect_stats:
        gate("2 scaler corpus", s is not None and s.group(1) == a.expect_stats, s.group(1) if s else "no scaler line (stats == dataset_name?)")
    else:
        print(f"  [info] scaler line: {s.group(1) if s else 'absent (stats == dataset_name)'}")
    w = re.search(r"\[warm-ring\] seeded torch\.Size\(\[(\d+), (\d+), (\d+)\]\) from (\S+)", out)
    med = re.search(r"\|oldest - newest\| over envs: median ([0-9.]+)", out)
    if a.warm:
        gate("3 warm-ring seeded", w is not None, w.group(0)[:100] if w else "no seeded line")
        gate("3b seed median > 0", med is not None and float(med.group(1)) > 0, med.group(0) if med else "")
        gate(f"3c {exp_idx}", exp_idx is not None and exp_idx in out)
    d = re.search(r"\[dtg-mpcc\] reference\.stride=(\d+) diff\.stride=(\d+) diff\.K=(\d+) diff\.H=(\d+)", out)
    gate(f"4 [dtg-mpcc] stride line == {S}/{S}", d is not None and d.group(1) == str(S) and d.group(2) == str(S), d.group(0) if d else "no line")
    meta_p = os.path.join(a.run, "meta.json")
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        st = meta.get("solver", {}).get("reference", {}).get("stride")
        gate(f"4b meta stride == {S}", st == S, f"meta stride={st}")
        ck = str(meta.get("diffusion", {}).get("ckpt_path"))
        if a.expect_dtg: gate("4c meta ckpt == expected", ck == a.expect_dtg, ck)
        else: print(f"  [info] meta ckpt: {ck}")
        pol = str(meta.get("policy", {}).get("_target_"))
        gate("4d meta policy is a warm class" if a.warm else "4d meta policy", ("Warm" in pol) if a.warm else True, pol)
        if a.cand:
            c = re.search(r"\[cand\] mode=(\S+) library=(\S+) .*gen_stride=(\d+) wp_per_cem_step=(\d+)", out)
            wp = str(5 // S)
            gate(f"4e [cand] gen_stride={S} wp={wp}", c is not None and c.group(3) == str(S) and c.group(4) == wp, c.group(0)[:120] if c else "no [cand] line")
    else:
        gate("4b meta.json present", False, meta_p)
    for f in ("results.json", "rollout_traj.npz", "control_ts_trace.npz"):
        gate(f"5 {f}", os.path.exists(os.path.join(a.run, f)))
    if os.path.exists(os.path.join(a.run, "results.json")):
        r = json.load(open(os.path.join(a.run, "results.json")))
        gate("5b per_env_success_held in results.json", "per_env_success_held" in r or "success_full" in r)
    print("VERDICT:", "PASS" if ok_all else "FAIL", a.run)
    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()
