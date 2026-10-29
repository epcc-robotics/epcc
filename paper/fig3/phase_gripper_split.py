#!/usr/bin/env python3
"""Phase split with the GRASP boundary from the EXPLICITLY RECORDED gripper
signal (not the cube-lift proxy or oracle kinematic thresholds).

Boundaries:
  t1 xy-align   : ||block_xy - eff_xy|| <= 0.04          (recorded positions)
  t_close GRASP : recorded gripper close transition       <-- EXPLICIT EVENT
  t_lift        : cube leaves table (cube_z - table_z > 0.02)   (physical event)
  t_above       : eff_z > 0.16
  t5 tgt-xy     : ||goal_xy - block_xy|| <= 0.04
  t6 success
Phases: above(xy) | descend | grasp(close->off-table) | lift | carry | place.
gripper transition detected per-signal via its own midpoint crossing (scale &
direction robust) -> works for demo (gripper_opening) and rollout (gripper).
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import stable_worldmodel as swm  # noqa
from latent_modules.eval.demo_ttc import compute_demo_ttc_for_run, _episode_state_seq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "baseline"))
from paper_style import paper_style, save_fig, series_color, series_hatch, DOUBLE_COL  # noqa: E402

REPRO = Path(__file__).resolve().parents[2]
RUN = REPRO / "findings/260521_baseline_setup/logs_dist_shift_abl/cube_dtg_base_reverify"
OUT = REPRO / "findings/260521_baseline_setup/ablation_b"
DS = "cube_multiview_diffusion_c0.3_postpad_fullscale"
TASK = "cube-mv"; CACHE = os.environ.get("STABLEWM_HOME")

P = ["above", "descend", "grasp", "lift", "carry", "place"]
LABELS = ["above\n(xy-align)", "descend\nto cube", "grasp\n(gripper)", "lift", "carry", "place\n(descend)"]
IS_CONTACT = [False, False, True, False, False, True]


def gripper_close_step(g):
    """First step where the gripper crosses its own midpoint into the CLOSED
    state — robust to scale and to open-high/close-low vs open-low/close-high."""
    g = np.asarray(g, float)
    lo, hi = np.nanmin(g), np.nanmax(g)
    if hi - lo < 0.05:
        return -1
    mid = 0.5 * (lo + hi)
    if g[0] < mid:                      # starts open(low) -> closes(high)
        idx = np.argmax(g > mid)
    else:                               # starts open(high) -> closes(low)
        idx = np.argmax(g < mid)
    return int(idx) if idx > 0 else -1


def fb(a, thr, start=0):
    for t in range(start, len(a)):
        if a[t] < thr:
            return t
    return -1


def fa(a, thr, start=0):
    for t in range(start, len(a)):
        if a[t] > thr:
            return t
    return -1


def lift_step(lift, thr=0.02, k=3):
    run = 0
    for t, b in enumerate(np.asarray(lift) > thr):
        run = run + 1 if b else 0
        if run >= k:
            return t - k + 1
    return -1


def events(eff, cube, goal, tz, grip, tsucc):
    T = int(tsucc)
    e = eff[:T + 1]; c = cube[:T + 1]; g = np.asarray(grip)[:T + 1]
    bxy = np.linalg.norm(e[:, :2] - c[:, :2], axis=1)
    lift = c[:, 2] - tz
    effz = e[:, 2]
    txy = np.linalg.norm(np.asarray(goal)[:2] - c[:, :2], axis=1)
    t1 = fb(bxy, 0.04)
    tc = gripper_close_step(g)
    tl = lift_step(lift)
    t4 = fa(effz, 0.16, start=max(tl, 0)) if tl >= 0 else -1
    t5 = fb(txy, 0.04, start=max(t4, 0)) if t4 >= 0 else -1
    return dict(t1=t1, tc=tc, tl=tl, t4=t4, t5=t5, t6=T)


def med_iqr(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    return (np.median(x), np.percentile(x, 25), np.percentile(x, 75)) if x.size else (np.nan,) * 3


def main():
    z = np.load(RUN / "rollout_traj.npz")
    epi = np.asarray(z["episode_idx"], np.int64); s0 = np.asarray(z["start_step"], np.int64)
    succ = np.asarray(z["success_full"]).astype(bool); fss = np.asarray(z["first_success_step"], float)
    eff = z["effector_pos"]; cube = z["cube_pos"]; goal = z["goal_pos"]
    tzc = np.asarray(z["table_z"], float); grip = z["gripper"]
    demo_ttc = compute_demo_ttc_for_run(RUN, criterion="gtr", force=False, cache_dir=CACHE)
    ds = swm.data.HDF5Dataset(DS, keys_to_cache=[], cache_dir=CACHE)

    a_ph = {p: [] for p in P}; reaches = []; dropped = 0; demo_grip_ok = 0; demo_grip_missing = 0
    verified = []
    for i in range(len(epi)):
        if not (succ[i] and fss[i] > 3 and demo_ttc[i] > 3):
            continue
        seq = _episode_state_seq(ds.load_episode(int(epi[i])), TASK)
        a = int(s0[i]); D = int(demo_ttc[i]); win = seq[a:a + D + 1]
        if len(win) < 4 or "cube_pos" not in win[0] or "effector_pos" not in win[0]:
            continue
        if "gripper_opening" not in win[0]:
            demo_grip_missing += 1; continue
        de = np.array([np.asarray(s["effector_pos"]) for s in win])
        dc = np.array([np.asarray(s["cube_pos"]) for s in win])
        dgrip = np.array([float(s["gripper_opening"]) for s in win])
        dg = np.asarray(win[0].get("goal_pos", goal[i])); dtz = float(win[0].get("table_z", dc[0, 2]))
        # verify demo gripper transition exists
        if len(verified) < 3:
            verified.append((float(dgrip.min()), float(dgrip.max()), gripper_close_step(dgrip), D))
        ed = events(de, dc, dg, dtz, dgrip, D)
        eo = events(eff[:, i, :], cube[:, i, :], goal[i], tzc[i], grip[:, i], fss[i])
        ks = ("t1", "tc", "tl", "t4", "t5")
        if any(ed[k] <= 0 for k in ks) or any(eo[k] <= 0 for k in ks):
            dropped += 1; continue
        demo_grip_ok += 1
        segd = dict(above=ed["t1"], descend=ed["tc"] - ed["t1"], grasp=ed["tl"] - ed["tc"],
                    lift=ed["t4"] - ed["tl"], carry=ed["t5"] - ed["t4"], place=ed["t6"] - ed["t5"])
        sego = dict(above=eo["t1"], descend=eo["tc"] - eo["t1"], grasp=eo["tl"] - eo["tc"],
                    lift=eo["t4"] - eo["tl"], carry=eo["t5"] - eo["t4"], place=eo["t6"] - eo["t5"])
        if any(segd[p] <= 0 or sego[p] <= 0 for p in P):
            dropped += 1; continue
        for p in P:
            a_ph[p].append(segd[p] / sego[p])
        reaches.append(float(np.linalg.norm(eff[0, i, :] - cube[0, i, :])))

    print("=== demo gripper signal check (min,max,close-step,demoLen) ===")
    for v in verified:
        print("  ", v)
    print(f"n usable = {len(a_ph['above'])}  dropped={dropped}  demo_grip_missing={demo_grip_missing}")
    print("=== 6-phase alpha, grasp via RECORDED gripper (median [IQR]) ===")
    for p in P:
        m, lo, hi = med_iqr(a_ph[p])
        print(f"  {p:8s}: {m:.2f} [{lo:.2f},{hi:.2f}]")
    mot = [v for p in P if not IS_CONTACT[P.index(p)] for v in a_ph[p]]
    con = [v for p in P if IS_CONTACT[P.index(p)] for v in a_ph[p]]
    print(f"MOTION median={np.median(mot):.2f}  CONTACT(grasp/place) median={np.median(con):.2f}")

    # ---- styled plot ----
    n = len(a_ph["above"])
    if n < 5:
        print("[plot] too few episodes; skipping"); return
    meds = np.array([med_iqr(a_ph[p])[0] for p in P])
    q1 = np.array([med_iqr(a_ph[p])[1] for p in P]); q3 = np.array([med_iqr(a_ph[p])[2] for p in P])
    order = np.argsort(reaches)[::-1]
    mat = np.array([a_ph[p] for p in P]).T[order]
    overall = np.median([v for p in P for v in a_ph[p]])
    gi = P.index("grasp")
    ours = series_color(0)   # single-method plot -> one series color, no hatch/gradient
    with paper_style():
        fig, (axb, axs) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.7),
                                       gridspec_kw={"width_ratios": [1.7, 1]})
        x = np.arange(len(P))
        axb.bar(x, meds, color=ours, edgecolor="black", linewidth=0.5)
        axb.errorbar(x, meds, yerr=[meds - q1, q3 - meds], fmt="none", ecolor="black",
                     elinewidth=0.8, capsize=2.5)
        for xi, m in zip(x, meds):
            axb.annotate(f"{m:.2f}", (xi, m), ha="center", va="bottom", fontsize=7,
                         xytext=(0, 1), textcoords="offset points")
        axb.axhline(1.0, ls="--", color="0.4", linewidth=0.8)
        # grasp = the ONE phase where our controller measurably slows (arm nearly
        # stationary while the gripper closes). No claim about the other phases.
        axb.annotate("arm dwells\n(gripper close)", (gi, meds[gi]),
                     xytext=(gi - 0.1, meds[gi] + 0.9), ha="center", va="bottom",
                     fontsize=6.5, color="0.25",
                     arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35"))
        axb.set_xticks(x); axb.set_xticklabels(["above\n(xy)", "descend", "grasp", "lift", "carry", "place"], fontsize=7)
        axb.set_ylabel(r"speedup $\alpha$ = demo / ours")
        axb.set_ylim(0, meds.max() * 1.6)
        axb.set_title("Phase-dependent speedup (no forced grouping)")
        snorm = TwoSlopeNorm(vcenter=round(overall, 2), vmin=1.0, vmax=2.6)
        im = axs.imshow(mat, aspect="auto", cmap="RdBu_r", norm=snorm,
                        interpolation="nearest", extent=[0, len(P), len(mat), 0])
        for xb in range(1, len(P)):
            axs.axvline(xb, color="white", linewidth=0.8)
        axs.set_xticks(np.arange(len(P)) + 0.5)
        axs.set_xticklabels(["abv", "dsc", "grp", "lft", "car", "plc"], fontsize=7)
        axs.set_ylabel("episode (reach-sorted)"); axs.set_title("per-episode"); axs.grid(False)
        fig.colorbar(im, ax=axs, fraction=0.06, pad=0.04).set_label(r"$\alpha$", rotation=0, labelpad=6)
        fig.suptitle("Speedup varies by phase; grasp is the one controller-side dwell",
                     fontsize=10, fontweight="bold")
    p = OUT / "ablB_phase6_profile.png"
    save_fig(fig, p); plt.close(fig)
    print(f"[plot] wrote {p}")


if __name__ == "__main__":
    main()
