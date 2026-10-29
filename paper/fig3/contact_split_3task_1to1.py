# FINAL-TREE VARIANT (2026-09-08) of the snapshot contact_split_3task.py: only paths/run maps changed.
# EPCC run dirs come from env EPCC_PUSHT / EPCC_CUBE / EPCC_SLIP (absolute paths); defaults = published rows.
# imports phase_compare_{pusht,cube}_final; writes fig_phase/ + *_values.json (per-panel medians)
#!/usr/bin/env python3
"""One figure, three tasks: speed-up in contact vs free motion.

The paper's claim is about contact-rich execution, so the main figure carries the two-way
split and nothing finer. PushT already splits that way natively (the agent is either
touching the T or travelling to it). Cube is collapsed from its six ordered phases onto
the same axis, using the same physical definition -- is the robot in contact with the
object -- so the three panels mean the same thing:

    contact      cube: grasp + lift + carry + place   pushT: agent touching the T
    free motion  cube: above + descend                pushT: travelling

alpha is a ratio of SUMMED times, per episode: (demo seconds in that group) / (method
seconds in that group). Nothing pairs the k-th contact bout with the k-th demo bout --
the counts differ -- which is why the sum, not a per-bout match, is the right object.

Both cube mappings are printed. The alternative ("precision only": contact = grasp +
place, with lift and carry counted as transport of an already-secured object) is a
defensible reading of contact-rich and it changes the cube panel, so it is reported
rather than silently discarded.

SAIL is k=3 only, repeat r1 only -- the repeats draw different episode sets, so pooling
them would score SAIL on a different population than the other rows.
"""
import sys
from pathlib import Path

import h5py, hdf5plugin  # noqa
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "scripts/baseline"))
import park_metrics as pm  # noqa
import stable_worldmodel as swm  # noqa
from latent_modules.eval.demo_ttc import compute_demo_ttc_for_run, _episode_state_seq

# ---------------------------------------------------------------- 1:1 sizing --
# Authored at the size it is PLACED at (subfigure 0.495\textwidth of IEEEtran's
# 7.16 in text width = 3.54 in), with every font at the caption's 8 pt, so
# \includegraphics[width=\linewidth] applies scale 1.0 and the in-figure text
# matches the caption exactly. The _final variant is authored at DOUBLE_COL=6.9 in
# and is then squeezed to 0.51x in the PDF, which is why its labels read ~4.6 pt.
import os as _os, matplotlib as _mpl
FIG_W = float(_os.environ.get("FIG_W", 3.54))
FIG_H = float(_os.environ.get("FIG_H", 2.55))
PT    = float(_os.environ.get("FIG_PT", 8))       # = IEEEtran caption size
_RC1TO1 = {"font.size": PT, "axes.titlesize": PT, "axes.labelsize": PT,
           "xtick.labelsize": PT, "ytick.labelsize": PT, "legend.fontsize": PT,
           "figure.dpi": 400, "savefig.dpi": 400}
from paper_style import (paper_style, save_fig, series_color, sparse_ticks,  # noqa
                         DOUBLE_COL)
from phase_gripper_split import events, P  # noqa
from phase_compare_pusht_final import TASKS, HOLD, DEMO_CT  # noqa
from phase_contact_pusht import dist_to_tee, R_AGENT, calibrate  # noqa
from phase_compare_cube_final import _ct_trace, METHODS as CUBE_METHODS, DS, TASK, CACHE  # noqa

OUT = REPO/"findings/260907_final_dtg_s1_warm" / "fig_phase"
ORDER = ["EPCC", "SuP", "SAIL", "SkiP"]
GROUPS = ["contact", "free motion"]
# Three defensible ways to call a cube phase "contact". They are printed together
# because the choice moves the cube panel, and CUBE_MAP picks the one that gets drawn.
CUBE_MAPS = {
    # the robot is physically touching the cube -- the same test pushT uses
    "hold":      {"grasp", "lift", "carry", "place"},
    # contact that has to be made or broken; lift/carry transport an already-secured cube
    "precision": {"grasp", "place"},
    # the single phase where the grip itself is being established
    "grasp":     {"grasp"},
}
CUBE_MAP = os.environ.get("CUBE_MAP", "hold")


def pick(methods):
    """Method list reduced to the four rows, SAIL at k=3 and its r1 only."""
    out = []
    for lab, runs in methods:
        rl = runs if isinstance(runs, (list, tuple)) else [runs]
        if "k{=}2" in lab or "k=2" in lab:
            continue
        name = ("SAIL" if "SAIL" in lab else
                "EPCC" if "ours" in lab or "EPCC" in lab else
                "SkiP" if "SkiP" in lab else "SuP")
        out.append((name, rl[:1]))
    return [(n, r) for n, r in sorted(out, key=lambda t: ORDER.index(t[0]))]


def pusht_alpha(task):
    cfg = TASKS[task]
    tol, agree, pf, tf = calibrate(cfg["demo"])
    print(f"  contact tol {tol:+.2f}px (agrees with the simulator on {agree:.1%})", flush=True)
    THR = R_AGENT + tol
    with h5py.File(cfg["demo"], "r") as f:
        ep_off = f["ep_offset"][:].astype(int)
        dbp, dap = f["block_pose"][:], f["pos_agent"][:]
    dframe = pm.demo_park_time(str(cfg["demo"]), hold_k=HOLD, mode="first", ttc_at="complete")
    per = {}
    for label, runs in pick(cfg["methods"]):
        acc = {}
        for run in runs:
            m = pm.park_metrics(run, hold_k=HOLD, mode="first", ttc_at="complete")
            if m is None:
                continue
            z = np.load(run / "rollout_traj.npz")
            bpr, ag = z["block_pose"], z["agent_pos"]
            T, n = bpr.shape[0], bpr.shape[1]
            cum = pm._time_axis(run, T, n,
                                float(m["results"].get("control_ts_mean", DEMO_CT)), m["results"])
            for i in range(n):
                if not m["success"][i]:
                    continue
                ep, st = int(m["episode_idx"][i]), int(m["start_step"][i])
                df = dframe[ep]
                if df < 0 or df - st < 4:
                    continue
                end = int(min(m["park_start"][i] + HOLD, T - 1))
                if end < 6:
                    continue
                o = ep_off[ep]
                dsl = slice(o + st, o + int(df) + 1)
                dcon = dist_to_tee(dap[dsl], dbp[dsl, :2], dbp[dsl, 2]) <= THR
                ocon = dist_to_tee(ag[:end+1, i, :], bpr[:end+1, i, :2], bpr[:end+1, i, 2]) <= THR
                dt = np.full(dcon.shape, DEMO_CT)
                ostep = np.diff(cum[:end+2, i])[:len(ocon)]
                d_c, d_f = float(dt[dcon].sum()), float(dt[~dcon].sum())
                o_c, o_f = (float(ostep[ocon[:len(ostep)]].sum()),
                            float(ostep[~ocon[:len(ostep)]].sum()))
                if min(d_c, d_f, o_c, o_f) <= 0:
                    continue
                acc[(ep, st)] = {"contact": d_c / o_c, "free motion": d_f / o_f}
        if acc:
            per[label] = acc
    return per


def cube_phase_times():
    """-> {method: {episode: {phase: (demo seconds, method seconds)}}}.

    Phase times are extracted once and grouped afterwards: the extraction dominates the
    runtime, and running it per mapping would triple it for no new data.
    """
    ds = swm.data.HDF5Dataset(DS, keys_to_cache=[], cache_dir=CACHE)
    seqc, per = {}, {}
    import yaml
    for label, runs in pick(CUBE_METHODS):
        acc = {}
        for run in runs:
            z = np.load(run / "rollout_traj.npz")
            epi = np.asarray(z["episode_idx"], np.int64); s0 = np.asarray(z["start_step"], np.int64)
            succ = np.asarray(z["success_full"]).astype(bool)
            fss = np.asarray(z["first_success_step"], float)
            eff, cube, goal = z["effector_pos"], z["cube_pos"], z["goal_pos"]
            tzc = np.asarray(z["table_z"], float); grip = z["gripper"]
            T, n = eff.shape[0], eff.shape[1]
            ct = _ct_trace(run, T, n)
            yaml.safe_dump(dict(task=TASK, eval_dataset=DS,
                                episode_idx=[int(x) for x in epi],
                                start_step=[int(x) for x in s0], cache_dir=str(CACHE)),
                           open(run / "meta.yaml", "w"))
            dtg = compute_demo_ttc_for_run(run, criterion="gtr", force=False, cache_dir=CACHE)
            for i in range(n):
                if not (succ[i] and fss[i] > 3 and dtg[i] > 3):
                    continue
                ep = int(epi[i])
                if ep not in seqc:
                    seqc[ep] = _episode_state_seq(ds.load_episode(ep), TASK)
                seq = seqc[ep]; a, D = int(s0[i]), int(dtg[i]); win = seq[a:a+D+1]
                if len(win) < 4 or "gripper_opening" not in win[0]:
                    continue
                de_ = np.array([np.asarray(s["effector_pos"]) for s in win])
                dc = np.array([np.asarray(s["cube_pos"]) for s in win])
                dgr = np.array([float(s["gripper_opening"]) for s in win])
                ed = events(de_, dc, np.asarray(win[0].get("goal_pos", goal[i])),
                            float(win[0].get("table_z", dc[0, 2])), dgr, D)
                eo = events(eff[:, i, :], cube[:, i, :], goal[i], tzc[i], grip[:, i], fss[i])
                kk = ("t1", "tc", "tl", "t4", "t5")
                if any(ed[k] <= 0 for k in kk) or any(eo[k] <= 0 for k in kk):
                    continue
                db = [0, ed["t1"], ed["tc"], ed["tl"], ed["t4"], ed["t5"], ed["t6"]]
                ob = [0, eo["t1"], eo["tc"], eo["tl"], eo["t4"], eo["t5"], eo["t6"]]
                if any(db[j+1]-db[j] <= 0 or ob[j+1]-ob[j] <= 0 for j in range(6)):
                    continue
                row = {}
                for j, ph in enumerate(P):
                    lo, hi = int(ob[j]), min(int(ob[j+1]), T)
                    row[ph] = ((db[j+1] - db[j]) * 0.05, float(ct[lo:hi, i].sum()))
                if min(v for pair in row.values() for v in pair) <= 0:
                    continue
                acc[ep] = row
        if acc:
            per[label] = acc
    return per


def cube_alpha(phase_times, contact_set):
    out = {}
    for label, eps in phase_times.items():
        acc = {}
        for ep, row in eps.items():
            tot = {"contact": [0.0, 0.0], "free motion": [0.0, 0.0]}
            for ph, (d, o) in row.items():
                g = "contact" if ph in contact_set else "free motion"
                tot[g][0] += d; tot[g][1] += o
            if min(v for gv in tot.values() for v in gv) <= 0:
                continue
            acc[ep] = {g: tot[g][0] / tot[g][1] for g in GROUPS}
        if acc:
            out[label] = acc
    return out


def paired(per):
    """Median per group over the episodes EVERY method solved."""
    labs = [l for l in ORDER if l in per]
    common = set.intersection(*[set(per[l]) for l in labs]) if labs else set()
    return ({l: {g: float(np.median([per[l][e][g] for e in common])) for g in GROUPS}
             for l in labs}, len(common), labs)


def main():
    panels = []
    for task in ("plain", "slippery"):
        name = {"plain": "PushT", "slippery": "Slippery-PushT"}[task]
        print(f"\n########## {name}", flush=True)
        res, ncom, labs = paired(pusht_alpha(task))
        print(f"  paired on {ncom} episodes every method solved")
        for l in labs:
            print(f"    {l:<14} contact {res[l]['contact']:.2f}   free {res[l]['free motion']:.2f}")
        panels.append((name, res, labs, ncom))

    print("\n########## Cube", flush=True)
    pt = cube_phase_times()
    drawn = None
    for key, cset in CUBE_MAPS.items():
        res, ncom, labs = paired(cube_alpha(pt, cset))
        mark = "  <== drawn" if key == CUBE_MAP else ""
        print(f"  [{key}: contact = {'+'.join(sorted(cset))}] paired on {ncom} episodes{mark}")
        for l in labs:
            print(f"    {l:<14} contact {res[l]['contact']:.2f}   "
                  f"free {res[l]['free motion']:.2f}   ratio {res[l]['contact']/res[l]['free motion']:.2f}")
        if key == CUBE_MAP:
            drawn = ("Cube", res, labs, ncom)
    # Column order matches Table I/II and Fig. 4: PushT, Cube, Slippery-PushT.
    panels.insert(1, drawn)

    with paper_style(), _mpl.rc_context(_RC1TO1):
        fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), constrained_layout=True)
        # "Slippery-PushT" is wider than its axes and constrained_layout sizes columns
        # from the axes, not the title, so the last title runs off the canvas. Pull the
        # axes area in from the right edge; the title then centres inside the figure.
        try:
            fig.get_layout_engine().set(rect=(0.0, 0.0,
                                              float(_os.environ.get("RIGHT_RECT", 1.0)), 1.0))
        except Exception as _e:
            print(f"[layout] rect not applied ({_e}); title may clip", flush=True)
        nm = len(ORDER)
        for ax, (name, res, labs, ncom) in zip(axes, panels):
            x = np.arange(len(GROUPS)); w = 0.8 / nm
            for j, l in enumerate(ORDER):
                if l not in res:
                    continue
                ax.bar(x + (j - (nm - 1) / 2) * w, [res[l][g] for g in GROUPS], w,
                       color=series_color(j, nm), edgecolor="black", linewidth=0.5,
                       zorder=3, label=l if name == panels[0][0] else None)
            ax.axhline(1.0, ls="--", color="0.45", lw=0.8, zorder=2)
            ax.set_xticks(x, ["contact", "free"], fontsize=PT)   # "free motion" does not clear
            ax.set_xlim(-0.45, 1.45)                            # "contact" at 8 pt in 1.05 in
            ax.tick_params(pad=1.5)
            ax.set_title(name, fontsize=PT, fontweight="bold", pad=3)
            top = max(v for m in res.values() for v in m.values())
            ax.set_ylim(0, top * 1.12)
            sparse_ticks(ax, "y", 4)
        axes[0].set_ylabel(r"speed-up $\alpha$", labelpad=1.5)
        xl = _os.environ.get("PHASE_XLAB", "")
        if xl:
            axes[1].set_xlabel(xl)
        h, l = axes[0].get_legend_handles_labels()
        if _os.environ.get("NOLEG", "0") != "1":
            fig.legend(h, l, loc="outside lower center", ncol=int(_os.environ.get("LEG_NCOL", 4)), fontsize=PT, fancybox=True,
                       edgecolor="0.5", framealpha=0.9, handletextpad=0.3, columnspacing=0.6,
                       borderpad=0.25, handlelength=1.2)   # LEG_NCOL=2 falls back to two rows
    suffix = "" if CUBE_MAP == "hold" else f"_{CUBE_MAP}"
    OUT.mkdir(parents=True, exist_ok=True)
    import json as _json
    (OUT / f"contact_split_3task{suffix}_values.json").write_text(_json.dumps(
        [{"panel": n, "paired_n": ncom, "alpha": res} for n, res, labs, ncom in panels], indent=1))
    fig.savefig(OUT / f"contact_split_3task{suffix}_1to1{_os.environ.get('SUF','')}.png", dpi=400)
    print("\nwrote", OUT / f"contact_split_3task{suffix}_1to1{_os.environ.get('SUF','')}.png", flush=True)


if __name__ == "__main__":
    main()
