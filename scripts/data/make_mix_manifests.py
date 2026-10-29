"""Generate the 6 aggregation mix manifests (Main1/Main2 x 3 tasks).

speed_c convention (checklist X3): native pace of each source's actions —
  O / D: the collection slowdown_factor (D files carry it in attrs)
  B_k:   k x the SuP backbone's demo pace (plain DP trained on 0.25-pace
         demos, slippery on 0.50, cube on 0.30)
"""
import glob
import json
import os
from pathlib import Path

H = Path(os.environ["STABLEWM_HOME"])
OUT = Path(__file__).resolve().parent / "config/mix"
OUT.mkdir(parents=True, exist_ok=True)

O_SRC = {
    "slippery": dict(path=str(H / "slippery_pusht/data/slippery_pusht_wm_corpus_excised.h5"),
                     c=1.0, name="O_excised", damping=0.35),
    "plain": dict(path=str(H / "datasets/pusht_expert_train_excised.h5"),
                  c=1.0, name="O_excised"),
    "cube": dict(path=str(H / "cube/collect/cube_multiview_WM.h5"),
                 c=1.0, name="O"),
}
B_DIR = {"slippery": "slippery", "plain": "pusht", "cube": "cube"}
DEMO_PACE = {"slippery": 0.5, "plain": 0.25, "cube": 0.3}
D_CS = {"slippery": ["0.5", "1.5", "2.5"], "plain": ["0.5", "1.5", "2.5"],
        "cube": ["0.3", "0.75", "1.2"]}

for task in ("slippery", "plain", "cube"):
    b_files = sorted(glob.glob(str(H / f"bset/{B_DIR[task]}/*_bset_k*_r*.h5")))
    b_files = [f for f in b_files if not f.endswith("oldcut_n50")]
    b_srcs = []
    for f in b_files:
        stem = Path(f).stem                      # {t}_bset_k{K}_r{R}
        k = int(stem.split("_k")[1].split("_")[0])
        src = dict(path=f, c=round(DEMO_PACE[task] * k, 3), name=f"B_k{k}_" + stem.split("_")[-1])
        if task == "slippery":
            src["damping"] = 0.35
        b_srcs.append(src)
    d_srcs = []
    for c in D_CS[task]:
        src = dict(path=str(H / f"dgrid/{task}/D_c{c}_final.h5"), c=float(c), name=f"D_c{c}")
        if task == "slippery":
            src["damping"] = 0.35
        d_srcs.append(src)

    main1 = [dict(O_SRC[task])] + b_srcs
    main2 = [dict(O_SRC[task])] + b_srcs + d_srcs
    for src in main1 + main2:                     # store paths relative to $STABLEWM_HOME
        src["path"] = os.path.relpath(src["path"], H)
    (OUT / f"{task}_main1.json").write_text(json.dumps(main1, indent=1))
    (OUT / f"{task}_main2.json").write_text(json.dumps(main2, indent=1))
    print(f"{task}: B files {len(b_srcs)}, main1 sources {len(main1)}, "
          f"main2 sources {len(main2)}")
print(f"wrote manifests to {OUT}")
