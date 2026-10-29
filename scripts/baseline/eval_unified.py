"""Phase 3 in-process unified eval entrypoint.

Single Python process per cell. No subprocess. No regex parsing. Metrics
hook directly to ``world.step()`` each iteration. Same loop body for every
policy — only ``cfg.policy._target_`` changes.

Architecture:
  1. ``setup_world_and_dataset(cfg)`` builds world + dataset (legacy helper).
  2. ``hydra.utils.instantiate(cfg.policy)`` builds the TaskOrthogonalPolicy.
  3. ``policy.reset(world, cfg=cfg)`` builds inner model + planner.
  4. ``world.set_policy(policy)`` attaches the policy for ``world.step()``.
  5. ``reset_world_to_dataset(...)`` rewinds envs to dataset chunks.
  6. Per-step: extract behavioral info → world.infos.update → world.step() →
     metrics.update(world.infos, ...). Single loop, no branching on policy.
  7. ``metrics.finalize()`` → unified results.json (both loose + strict
     success_rate for cube; TTC for all).
"""
from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# latent_diffusion_cd package (for DTG variants) lives in the repository;
# add explicitly in case PYTHONPATH wasn't inherited (Hydra may alter env).
_LD_CD_ROOT = REPO_ROOT / "latent_modules" / "diffusion_cd"
if _LD_CD_ROOT.exists() and str(_LD_CD_ROOT) not in sys.path:
    sys.path.insert(0, str(_LD_CD_ROOT))

# Import _runtime first to register the Hydra legacy-eval search path.
import policy._runtime  # noqa: F401


def _output_dir(cfg: DictConfig) -> Path:
    # `cfg.output.logs_subdir` lets a recipe (or CLI override) redirect runs
    # away from the shared `logs_v3_inprocess/` tree without forking the
    # script. Defaults preserve historical behavior. Example override:
    #   python eval_inprocess.py ... ++output.logs_subdir=logs_v3_repro
    logs_subdir = str(cfg.output.get("logs_subdir", "logs_v3_inprocess"))
    base = REPO_ROOT / cfg.output.findings_dir / logs_subdir / cfg.output.run_name
    base.mkdir(parents=True, exist_ok=True)
    return base


def _save_results(metrics: dict, cfg: DictConfig, outdir: Path) -> None:
    """Write the unified v3 schema: results.json + results.txt + meta.json."""
    json_payload = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            json_payload[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating, np.bool_)):
            json_payload[k] = v.item()
        elif isinstance(v, (int, float, str, bool, list, dict, type(None))):
            json_payload[k] = v
    (outdir / "results.json").write_text(json.dumps(json_payload, indent=2, default=str) + "\n")

    sr_strict = float(metrics.get("success_rate", float("nan")))
    sr_loose = float(metrics.get("success_rate_loose", sr_strict))
    # Label is task-aware: cube `success_rate` = CubeGTRMetrics.success_full
    # (strict GTR = Grasped∧NoDrop∧Reached); PushT `success_rate` =
    # PushTCoverageMetrics block-only per_env_success — NOT a GTR criterion.
    _is_cube = (str(cfg.task.get("name", "")) == "cube") or bool(
        cfg.task.get("requires_wrist_patch", False))
    _is_scene = str(cfg.task.get("task_name", "")).lower() == "scene"
    _seg = metrics.get("segment")
    if _is_cube:
        _sr_label = "Success rate (strict GTR)"
    elif _is_scene and _seg:
        _sr_label = f"Success rate (segment {_seg}, physical)"
    elif _is_scene:
        _sr_label = "Success rate (terminated, provisional)"
    else:
        _sr_label = "Success rate (block-only)"
    txt = [
        f"{_sr_label}: {sr_strict*100:.1f}%",
    ]
    if _is_cube:
        txt.append(f"Success rate (loose):      {sr_loose*100:.1f}%")
    # Scene segment breakdown: which stage the rollouts actually die on is the
    # whole point of the segment sweep, so surface it in the headline file.
    if _is_scene and _seg:
        txt.append(f"Success rate (held {int(metrics.get('hold_window', 0))} steps): {float(metrics.get('success_rate_held_window', float('nan')))*100:.1f}%  <- hold criterion")
        txt.append(f"Success rate (held at end):{float(metrics.get('success_rate_held', float('nan')))*100:.1f}%  (artefact-prone: segment ref ends at the goal)")
        rates = metrics.get("stage_reached_rate") or {}
        for _s in str(_seg):
            if _s in rates:
                txt.append(f"  stage {_s} reached:        {float(rates[_s])*100:.1f}%")
        _degen = metrics.get("stage_true_at_start") or {}
        _bad = [s for s, v in _degen.items() if np.asarray(v).any()]
        if _bad:
            txt.append(f"  [WARN] stages already satisfied at t=0: {_bad}")
        txt.append(f"Success rate (CLEAN):      {float(metrics.get('success_rate_clean', float('nan')))*100:.1f}%  (held AND no collateral damage)")
        txt.append(f"  wrong_button (btn0 off):  {float(metrics.get('wrong_button_rate', float('nan')))*100:.1f}%  (locks the drawer — undoes A/C)")
        txt.append(f"  prior_stage_broken:       {float(metrics.get('prior_stage_broken_rate', float('nan')))*100:.1f}%  (lost a stage it was handed)")
        txt.append(f"full_physical_success:     {float(metrics.get('full_physical_success_rate', float('nan')))*100:.1f}%  (all 5 conditions at once)")
    txt += [
        f"n_envs:                    {metrics.get('n_envs', '?')}",
        f"run_name:                  {cfg.output.run_name}",
    ]
    # PushT-specific DP-paper-canonical aggregates (block-only sr above + coverage stats).
    if "mean_max_coverage" in metrics:
        txt.append(f"mean_max_coverage:         {float(metrics['mean_max_coverage']):.4f}  (DP-paper test/mean_score)")
    if "mean_final_coverage" in metrics:
        txt.append(f"mean_final_coverage:       {float(metrics['mean_final_coverage']):.4f}  (end-step coverage)")
    if "ttc_steps_mean" in metrics:
        txt.append(f"ttc_steps_mean:            {float(metrics['ttc_steps_mean']):.2f}")
    if "ttc_sim_time_mean_s" in metrics:
        txt.append(f"ttc_sim_time_s:            {float(metrics['ttc_sim_time_mean_s']):.3f}  (legacy: steps × recipe-dt)")
    if "ttc_sim_time_actual_mean_s" in metrics:
        txt.append(f"ttc_sim_time_actual_s:     {float(metrics['ttc_sim_time_actual_mean_s']):.3f}  (cumulative env dt — fair for SAIL)")
    if "control_timestep_s" in metrics:
        txt.append(f"control_timestep_s:        {float(metrics['control_timestep_s']):.5f}  (recipe nominal)")
    if "control_ts_min" in metrics and "control_ts_max" in metrics:
        ct_min = float(metrics["control_ts_min"]); ct_max = float(metrics["control_ts_max"])
        if abs(ct_max - ct_min) > 1e-9:
            txt.append(f"control_ts_range:          [{ct_min:.5f}, {ct_max:.5f}]  (mean={float(metrics['control_ts_mean']):.5f})")
    if "wall_time_s" in metrics:
        txt.append(f"wall_time_s:               {float(metrics['wall_time_s']):.1f}")
    if "mean_final_dist" in metrics:
        txt.append(f"mean_final_dist:           {float(metrics['mean_final_dist']):.4f}")
    (outdir / "results.txt").write_text("\n".join(txt) + "\n")

    (outdir / "meta.json").write_text(
        json.dumps(OmegaConf.to_container(cfg, resolve=False), indent=2, default=str) + "\n"
    )


def _annotate_frame(arr: np.ndarray, t: int, n_replans: int, replan_flag: bool) -> np.ndarray:
    """Burn `t=NNNN` (top-left) and `R=NN[*]` (top-right) on an HWC uint8 RGB frame.

    The `*` and red color highlight the frame where `n_replans` just incremented,
    so it's visually obvious which step triggered a replan. Black outline around
    white text keeps the overlay readable on any background.
    """
    from PIL import Image, ImageDraw  # lazy import — only paid when video on
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    w, _ = img.size
    t_text = f"t={t:04d}"
    r_text = f"R={n_replans:02d}" + ("*" if replan_flag else "")

    def _outlined(xy, text, fill):
        x, y = xy
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((x + dx, y + dy), text, fill="black")
        draw.text((x, y), text, fill=fill)

    _outlined((2, 2), t_text, "white")
    # Right-align R by approximating default-font width (~6 px/char).
    rx = max(2, w - 6 * len(r_text) - 2)
    _outlined((rx, 2), r_text, "red" if replan_flag else "white")
    return np.asarray(img, dtype=np.uint8)


_RATE_FONT: dict = {}


def _rate_font(px: int):
    """A scalable face at `px`, cached. PIL's built-in font is an 11-px bitmap.

    PushT renders at 96x96 (`resolution: 96` in the cell, and it is the policy's
    actual observation — raising it would change what the net sees, so the frame
    stays small and the video is upscaled instead). At 96 px the bitmap font is
    legible only after the whole frame is blown up, which blurs it. Drawing at
    the upscaled size with a real face keeps the label crisp.
    """
    if px not in _RATE_FONT:
        from PIL import ImageFont
        try:
            import matplotlib
            ttf = (Path(matplotlib.__file__).parent / "mpl-data" / "fonts"
                   / "ttf" / "DejaVuSans-Bold.ttf")
            _RATE_FONT[px] = ImageFont.truetype(str(ttf), px)
        except Exception:
            _RATE_FONT[px] = ImageFont.load_default()
    return _RATE_FONT[px]


def _annotate_rate(arr: np.ndarray, ct: float, base: float,
                   upscale: int = 1) -> np.ndarray:
    """Upscale an HWC uint8 frame and burn the AWE branch bottom-left.

    The slow branch runs at the recipe's base control period; the fast branch at
    base/k. So comparing the realised period against `base` recovers the branch
    without needing to know k.

    Applied at video-write time rather than at capture, for two reasons. The
    rate for step t is chosen inside get_action(), i.e. inside world.step(), so
    a label drawn at capture time would name the *previous* step's branch — the
    caller must pass the period read back after the step. And the upscale has to
    happen before the text, or the text is upscaled too and goes soft.
    """
    from PIL import Image, ImageDraw  # lazy import — only paid when video on
    slow = ct >= base * 0.99
    txt = "SLOW" if slow else (f"FAST x{base / ct:.1f}" if ct > 0 else "FAST")
    img = Image.fromarray(arr)
    if upscale > 1:
        # NEAREST: the source is a 96-px render, and a smooth filter turns the
        # already-burned t=/R= overlay into mush.
        img = img.resize((img.width * upscale, img.height * upscale),
                         Image.NEAREST)
    draw = ImageDraw.Draw(img)
    h = img.height
    fs = max(11, int(round(h * 0.075)))
    font = _rate_font(fs)
    pad = max(2, fs // 4)
    x, y = pad, h - fs - pad
    off = max(1, fs // 10)
    for dx in (-off, 0, off):
        for dy in (-off, 0, off):
            if dx or dy:
                draw.text((x + dx, y + dy), txt, fill="black", font=font)
    draw.text((x, y), txt, fill="#FF4040" if slow else "#40FF40", font=font)
    return np.asarray(img, dtype=np.uint8)


def _get_replan_count(policy) -> int:
    """Best-effort read of the policy's replan counter.

    Walks the policy tree (policy → _inner → _inner._inner / _solver /
    _inner_solver) and probes for ``n_replans`` / ``_n_replans``. Surfaces a
    meaningful counter for every policy family in the v2 loop:
      * DPPolicy                                — own ``n_replans``
      * ChunkedReplanPolicy (PushTDPRepoPolicy) — base class counter
      * WrapperPolicy (SAILPolicy*)             — forwards from ``_inner``
      * MPCCPolicy / DTGMPCCPolicy              — per-step CEM counter
      * DTGInvDynPolicy                         — per-step fallback (or legacy
                                                  inner planner's counter)
      * WMFullPolicy                            — stays 0 (single rollout)
    """
    nodes = [policy]
    # Walk one level into wrapper attributes commonly used in this codebase.
    for attr in ("_inner", "_solver"):
        node = getattr(policy, attr, None)
        if node is not None:
            nodes.append(node)
            for sub in ("_inner", "_solver", "_inner_solver"):
                sub_node = getattr(node, sub, None)
                if sub_node is not None:
                    nodes.append(sub_node)

    for node in nodes:
        for attr in ("n_replans", "_n_replans"):
            v = getattr(node, attr, None)
            if v is not None:
                return int(v)
    return 0


def _read_control_dt(world, n_envs: int, default_ct: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-env (requested, realised) sim-seconds advanced by the last step().

    Two axes, deliberately:

    * **requested** is what the policy asked for — `env._control_timestep`
      (OGBench) or `1/env.control_hz` (PushT). Every archived result was
      computed on this axis, so keeping it is what makes AWE-OFF runs
      reproduce.
    * **realised** is what MuJoCo actually advanced. On OGBench,
      `set_timesteps()` updates `_n_steps` immediately but `_model.opt.timestep`
      is written only inside `compile_model_and_data()`, which `reset()` runs
      only when the model is dirty — so a mid-rollout physics-dt request never
      reaches the solver and the true period is `_n_steps * opt.timestep`. For
      the AWE factors (fast 2.5 / slow 1.0) the quotients are exact integers on
      both tasks so the two axes agree; they diverge by up to ~4% at
      non-integer k, which is worth surfacing rather than silently averaging.

    PushT recomputes its substep count from instance attributes on every
    `step()`, so there the formula is exact by construction, not inferred.
    """
    req = np.empty(n_envs, dtype=np.float64)
    real = np.empty(n_envs, dtype=np.float64)
    envs = world.envs.unwrapped.envs
    for i in range(n_envs):
        ev = envs[i].unwrapped
        ct = getattr(ev, "_control_timestep", None)
        if ct is not None:                                   # OGBench ManipSpace
            req[i] = float(ct)
            nst = getattr(ev, "_n_steps", None)
            model = getattr(ev, "_model", None)
            real[i] = (float(nst) * float(model.opt.timestep)
                       if nst is not None and model is not None else float(ct))
            continue
        hz = getattr(ev, "control_hz", None)
        if hz:                                               # PushT2D
            req[i] = 1.0 / float(hz)
            dt = float(getattr(ev, "dt", 0.0))
            real[i] = (max(1, int(round(1.0 / (dt * float(hz))))) * dt) if dt > 0 else req[i]
            continue
        req[i] = real[i] = default_ct
    return req, real


# Config dir is overridable via EVAL_CONFIG_DIR so an alternate recipe folder
# (e.g. config/eval/paper) can be swapped in as the MAIN config dir
# without editing this file. Relative to this script (scripts/baseline/).
_EVAL_CONFIG_DIR = os.environ.get("EVAL_CONFIG_DIR", "../../config/eval/cells")


@hydra.main(version_base=None, config_path=_EVAL_CONFIG_DIR, config_name=None)
def main(cfg: DictConfig) -> None:
    if "STABLEWM_HOME" not in os.environ:
        raise RuntimeError("STABLEWM_HOME env var must point at the data root (see docs/DATA.md).")

    print("=" * 80)
    print(f"[scripts/baseline_v2/eval_inprocess.py] cell={cfg.output.run_name}")
    # task_name is the new single-source-of-truth discriminator
    # (cube | pusht | ...). Falls back to legacy success_spec_key for
    # baseline_corl_v2 recipes that haven't migrated yet.
    task_name = str(cfg.task.get("task_name") or cfg.task.get("success_spec_key", "")).lower()
    if task_name.startswith("cube"):
        task_name = "cube"
    elif task_name == "pusht":
        task_name = "pusht"
    elif task_name == "granular":
        task_name = "granular"
    print(f"  task: {cfg.task.env_name} (task_name={task_name})")
    print(f"  policy._target_: {cfg.policy._target_}")
    print(f"  metrics._target_: {cfg.task.metrics._target_}")
    print(f"  num_eval: {cfg.eval.num_eval}, seed: {cfg.seed}")
    # PyFleX has process-global C state — only one DeformableEnv can live
    # per Python process. Hard-fail early rather than silently corrupting
    # rollouts. See policy/granular_dp_dino.py for the same guard.
    if task_name == "granular" and int(cfg.eval.num_eval) > 1:
        raise RuntimeError(
            f"PyFleX single-env constraint: granular task requires num_eval=1, "
            f"got {cfg.eval.num_eval}. PyFleX has process-global C state — "
            f"multiple envs in one process clobber each other."
        )
    print("=" * 80)

    from policy._runtime import (
        setup_world_and_dataset, reset_world_to_dataset,
        extract_cube_behavioral, initial_table_z, extract_pusht_coverage,
        extract_scene_behavioral,
        patch_pusht_block_only_success, force_pusht_fixed_goal,
    )

    # Phase 1: world + dataset
    ns = setup_world_and_dataset(cfg)
    world = ns.world
    dataset = ns.dataset
    n_eval = ns.n_eval
    eval_budget = ns.eval_budget
    print(f"[eval_inprocess] n_eval={n_eval}, effective eval_budget={eval_budget} "
          f"(recipe says cfg.eval.eval_budget={cfg.eval.eval_budget}, full={cfg.get('full', False)})")

    # Block-only success patch for pusht. Stock swm/PushT-v1 eval_state uses a
    # 4-D L2 over [agent_xy, block_xy] which the expert demos themselves don't
    # satisfy (0/47 in expert_demo_025_success). Replace with block-only
    # success matching the canonical DP-paper convention + the 260509 baseline.
    # No-op for cube. See policy._runtime.patch_pusht_block_only_success.
    if task_name == "pusht":
        patch_pusht_block_only_success(world)
        print("[eval_inprocess] patched pusht env.eval_state -> block-only success "
              "(pos_dims=[2,3], pos<20, angle<pi/9)")
        # Force every reset to apply PUSHT_FIXED_GOAL. Necessary because
        # reset_world_to_dataset filters columns starting with "goal" out of
        # init_step, so a `_set_goal_state` callable receives a SYNTHESIZED
        # value (chunk's last state) rather than the real dataset goal — which
        # for random-start chunks leaks block-near-goal pairs and produces
        # spurious step-0 successes (see findings/260524).
        force_pusht_fixed_goal(world)
        print("[eval_inprocess] forced pusht goal_state -> PUSHT_FIXED_GOAL "
              "(256, 256, π/4) on every env.reset")

    # Granular: parallels the pusht block above. reset_world_to_dataset
    # filters columns starting with "goal" out of init_step, so the env's
    # internal goal_state must be re-applied after every reset from the
    # dataset's saved goal_state (which IS passed as options={'goal_state': ...}
    # on the first reset; force_granular_goal captures + re-applies it).
    if task_name == "granular":
        from policy._runtime import force_granular_goal
        force_granular_goal(world)
        print("[eval_inprocess] forced granular goal_state -> dataset goal_state "
              "on every env.reset")

    # Phase 2: policy instantiation + reset.
    # _recursive_=True so nested _target_ structures get built bottom-up
    # (SAILPolicy needs its `inner` DPPolicy instantiated first; DPPolicy
    # has no nested targets so recursive is a no-op for it).
    policy = hydra.utils.instantiate(cfg.policy, _recursive_=True)
    # TaskOrthogonalPolicy.reset accepts (world, cfg, eval_world) when the
    # policy needs to construct legacy helpers (DP, MPCC, …). Base no-op
    # accepts (world) only — gracefully degrade for stateless policies.
    try:
        policy.reset(world, cfg=cfg, eval_world=ns)
    except TypeError:
        try:
            policy.reset(world, cfg=cfg)
        except TypeError:
            policy.reset(world)
    world.set_policy(policy)

    # Capture the executed action each step so save_traj can dump the real
    # commanded action (else rollout_traj.npz['action'] is all-zeros — the
    # (n,1) default was never overwritten). Port of eval_mpc_multiview.py:684-697.
    if not getattr(world.policy, "_logger_action_capture_installed", False):
        _orig_get_action = world.policy.get_action

        def _capture_get_action(info_dict, __orig=_orig_get_action, **kw):
            out = __orig(info_dict, **kw)
            try:
                world.policy._last_action_executed = (
                    out.detach().cpu().numpy() if hasattr(out, "detach")
                    else np.asarray(out)
                )
            except Exception:
                world.policy._last_action_executed = None
            return out

        world.policy.get_action = _capture_get_action
        world.policy._logger_action_capture_installed = True

    # Mirror eval_cleandiffuser_dp.run() lines ~407-416: attach wrist renderer
    # + proprio aliasing if the model needs them. Without this, the policy
    # can't read multiview pixels / proprio from world.infos and predicts
    # garbage actions. Search up the wrapper chain — SAILPolicy holds the
    # model on its inner DPPolicy.
    inner_model = getattr(policy, "_model", None)
    _probe = policy
    while inner_model is None and hasattr(_probe, "_inner"):
        _probe = _probe._inner
        inner_model = getattr(_probe, "_model", None)
    if inner_model is not None:
        enabled_views = getattr(inner_model, "enabled_views", ("agent", "wrist"))
        proprio_keys = tuple(getattr(inner_model, "proprio_keys", ()) or ())
        if "wrist" in enabled_views:
            from eval_cleandiffuser_dp import _attach_wrist_renderers
            _attach_wrist_renderers(world, img_size=cfg.world.get("width", 224))
            print("[eval_inprocess] wrist renderer attached")
        if "proprio" in enabled_views:
            from eval_cleandiffuser_dp import _attach_proprio_alias
            _attach_proprio_alias(world, proprio_keys)
            print(f"[eval_inprocess] proprio alias attached (keys={proprio_keys})")

    # [Ablation C] init-state perturbation (pusht): gaussian noise on
    # [agent_xy, block_xy, theta] of the dataset init state. Fixed seed so
    # every cell (arm x N x radius) evaluates the SAME perturbed init set ->
    # paired comparisons. Off unless perturb_xy_std / perturb_theta_std > 0.
    _p_xy = float(cfg.eval.get("perturb_xy_std", 0.0) or 0.0)
    _p_th = float(cfg.eval.get("perturb_theta_std", 0.0) or 0.0)
    _p_blk = float(cfg.eval.get("perturb_block_xy_std", 0.0) or 0.0)  # cube, meters
    init_perturb_fn = None
    if _p_blk > 0.0:
        # CUBE: gaussian noise on the block freejoint xy (qpos[14:16] — verified
        # against privileged_block_0_pos on cube_multiview_diffusion postpad).
        # prev_qpos gets the SAME offset — the env restores prev_qpos, and a
        # mismatched pair would inject a finite-difference velocity spike at
        # frame 0 (the exact artifact the drift screen removes from datasets).
        if task_name != "cube":
            raise NotImplementedError("perturb_block_xy_std is cube-only")
        _p_seed = int(cfg.eval.get("perturb_seed", 777))
        _perturb_rec = {}

        def init_perturb_fn(init_step):
            rng = np.random.default_rng(_p_seed)
            q = np.asarray(init_step["qpos"], dtype=np.float64).copy()
            _perturb_rec["base"] = q.copy()
            off = rng.normal(0.0, _p_blk, size=(q.shape[0], 2))
            q[:, 14:16] += off
            q[:, 14] = np.clip(q[:, 14], 0.28, 0.60)   # table x
            q[:, 15] = np.clip(q[:, 15], -0.35, 0.05)  # table y
            init_step["qpos"] = q
            if "prev_qpos" in init_step:
                pq = np.asarray(init_step["prev_qpos"], dtype=np.float64).copy()
                pq[:, 14:16] = q[:, 14:16]             # same offset -> zero fd-velocity
                init_step["prev_qpos"] = pq
            _perturb_rec["perturbed"] = q.copy()
            print(f"[perturb] cube block xy noised: std={_p_blk}m seed={_p_seed} "
                  f"n={q.shape[0]} (qpos[14:16] + prev_qpos synced)")
            return init_step

    elif _p_xy > 0.0 or _p_th > 0.0:
        if task_name != "pusht":
            raise NotImplementedError(
                "eval.perturb_* implemented for pusht state layout only")
        _p_seed = int(cfg.eval.get("perturb_seed", 777))

        _perturb_rec = {}

        def init_perturb_fn(init_step):
            rng = np.random.default_rng(_p_seed)
            st = np.asarray(init_step["state"], dtype=np.float64).copy()
            _perturb_rec["base"] = st.copy()
            st[:, 0:4] += rng.normal(0.0, _p_xy, size=st[:, 0:4].shape)
            st[:, 0:4] = np.clip(st[:, 0:4], 50.0, 462.0)   # stay off the walls
            st[:, 4] += rng.normal(0.0, _p_th, size=st.shape[0])
            init_step["state"] = st
            _perturb_rec["perturbed"] = st.copy()
            print(f"[perturb] init noised: xy_std={_p_xy}px theta_std={_p_th}rad "
                  f"seed={_p_seed} n={st.shape[0]} (same draws in every cell)")
            return init_step

    # Phase 3: env reset to dataset chunk
    init_step, goal_step = reset_world_to_dataset(
        world,
        dataset,
        episode_indices=ns.eval_episodes,
        start_steps=ns.eval_start_idx,
        goal_offset_steps=(
            np.full(n_eval, int(ns.goal_offset), dtype=np.int64)
            if np.ndim(ns.goal_offsets) == 0
            else np.asarray(ns.goal_offsets, dtype=np.int64)
        ),
        callables=(
            OmegaConf.to_container(cfg.eval.callables, resolve=True)
            if cfg.eval.get("callables") else None
        ),
        init_perturb_fn=init_perturb_fn,
        verify_init=bool(cfg.eval.get("verify_init", False)),
    )

    # Phase 4: metric instantiation
    metrics = hydra.utils.instantiate(cfg.task.metrics, _recursive_=False)

    # Task-specific per-step behavioral capture preamble.
    # Cube branch: derived from task_name. Legacy fallback for unmigrated
    # baseline_corl_v2 recipes still on `requires_wrist_patch`.
    is_cube = (task_name == "cube") or bool(cfg.task.get("requires_wrist_patch", False))
    if is_cube:
        goal_pos_per_env = goal_step.get("goal_privileged_block_0_pos")
        if goal_pos_per_env is None:
            goal_pos_per_env = np.zeros((n_eval, 3))
        else:
            goal_pos_per_env = goal_pos_per_env[:, 0, ...] if goal_pos_per_env.ndim > 2 else goal_pos_per_env
        table_z_per_env = initial_table_z(world, n_eval)

    # Phase 5: unified env-step loop
    t0 = time.time()
    env_mask = np.ones(n_eval, dtype=bool)
    ever_terminated = np.zeros(n_eval, dtype=bool)   # loose success: any env.terminated at any step

    # Per-step actual control_timestep. For SAIL/SpeedAdaptive recipes the
    # policy mutates env._control_timestep mid-rollout, so the legacy
    # `ttc_steps * cfg.control_timestep_s` understates real speedup.
    # Capture the per-step dt here; sim-time TTC is recomputed post-finalize.
    # Each entry is an (n_envs,) vector, not a scalar: SAIL's AWE scheduler can
    # give every env its own rate (policy/sail.py precision_mode="per_env"), so
    # a single number per step would be a lie the moment the rates diverge.
    # `_real` is the period MuJoCo actually advanced — see _read_control_dt.
    control_ts_per_step: list[np.ndarray] = []
    control_ts_real_per_step: list[np.ndarray] = []
    _default_ct = float(cfg.task.get("control_timestep_s", 0.05))

    # Optional video capture — accumulate per-env frames, save MP4s at the end.
    debug_video = bool(cfg.eval.get("debug_video", False))
    video_frames: list[list[np.ndarray]] | None = (
        [[] for _ in range(n_eval)] if debug_video else None
    )
    # Tracks the policy's last-seen replan count so we can flag the exact frame
    # on which a replan fired (highlight in red on the overlay).
    prev_n_replans = _get_replan_count(policy) if debug_video else 0
    if debug_video:
        print(f"[eval_inprocess] debug_video=True — capturing per-env frames "
              f"(will save {n_eval} MP4s to {outdir if False else cfg.output.run_name}/videos/)")

    # Optional per-step trajectory dump (cube/effector/gripper + action) for
    # PRECISE phase segmentation (reach vs transport, grasp moment) of OUR
    # rollout — not just the demo. Toggle: eval.save_traj (cube only).
    _is_pusht_traj = (task_name == "pusht")
    _is_scene_traj = (task_name == "scene")
    save_traj = bool(cfg.eval.get("save_traj", False)) and (
        is_cube or _is_pusht_traj or _is_scene_traj)
    if save_traj and _is_scene_traj:
        # Scene had no trajectory logging at all. Per-step physical state is
        # what makes a stage failure diagnosable: `stage_first_step` says WHEN
        # a stage completed, but only the trace says what the drawer/cube/arm
        # were doing while it did not.
        traj = {"drawer_qpos": [], "window_qpos": [], "block_pos": [],
                "drawer_handle_y": [], "button0": [], "button1": [],
                "effector_pos": [], "gripper": [], "action": []}
    elif save_traj and is_cube:
        traj = {"cube_pos": [], "effector_pos": [], "gripper": [], "action": []}
    elif save_traj and _is_pusht_traj:
        # PushT root-cause logging: block/agent pose + agent vel + BLOCK vel
        # (linear+angular, for an EXACT env-fork restore) + executed action.
        traj = {"block_pose": [], "agent_pos": [], "vel_agent": [],
                "block_vel": [], "block_angvel": [], "action": []}
    else:
        traj = None
    if save_traj:
        print(f"[eval_inprocess] save_traj=True — dumping per-step "
              f"{'cube/eff/grip' if is_cube else 'block/agent/vel'}/action → rollout_traj.npz")

    # [state-fork] Full MuJoCo integration state per sub-env, per env-step, so an offline
    # replay can restore ANY mid-episode state into a fresh env and roll a DIFFERENT action
    # sequence from it. rollout_traj.npz cannot do this: it holds cube/effector/gripper only,
    # which is not a simulator state. Opt-in via ++eval.save_mj_state=true; a no-op on pymunk
    # PushT, which has no MjData.
    save_mj_state = bool(cfg.eval.get("save_mj_state", False))
    mj_states = []
    _mj_size = _mj_sig_int = 0
    _capture_mj_state = None
    if save_mj_state:
        import mujoco as _mjs
        _u0 = world.envs.unwrapped.envs[0].unwrapped
        if not hasattr(_u0, "_data"):
            print("[mj_state] env exposes no MjData — save_mj_state disabled")
            save_mj_state = False
        else:
            _mj_sig = _mjs.mjtState.mjSTATE_INTEGRATION
            _mj_sig_int = int(_mj_sig.value)
            _mj_size = int(_mjs.mj_stateSize(_u0._model, _mj_sig))
            print(f"[mj_state] mjSTATE_INTEGRATION size={_mj_size} float64/env "
                  f"-> {(eval_budget + 1) * n_eval * _mj_size * 8 / 1e6:.2f} MB raw")

            def _capture_mj_state():
                buf = np.zeros((n_eval, _mj_size), dtype=np.float64)
                for _i, _e in enumerate(world.envs.unwrapped.envs):
                    _u = _e.unwrapped
                    _mjs.mj_getState(_u._model, _u._data, buf[_i], _mj_sig)
                return buf

    # [Ablation C] policy-independent latent trace: every K env-steps, encode
    # the CURRENT observation with the task's WM encoder (same encode path /
    # pool='lewm' as the demo latent caches) -> theta* is measurable for ANY
    # policy. DP produces no latents of its own, and MPCC's cur_emb is
    # solve-granular + policy-internal, so the eval loop measures externally.
    # Toggle: +eval.latent_trace_every=5 (0/absent = off).
    _lt_every = int(cfg.eval.get("latent_trace_every", 0) or 0)
    lt_latents, lt_steps = [], []
    if _lt_every > 0:
        import importlib.util as _ilu
        import torch as _torch
        _cl_path = Path(__file__).resolve().parents[1] / "cache_latents.py"
        _spec = _ilu.spec_from_file_location("_cache_latents", str(_cl_path))
        _cl = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_cl)
        _lt_ckpt = cfg.eval.get("latent_trace_ckpt", None) or str(
            Path(os.environ["STABLEWM_HOME"]) / "checkpoints"
            / f"{cfg.task.policy_config_group}_object.ckpt")
        _lt_dev = "cuda" if _torch.cuda.is_available() else "cpu"
        _, _lt_enc, _lt_proj = _cl.load_lewm(_lt_ckpt, _lt_dev)
        _lt_mean, _lt_std = _cl.get_imagenet_stats(_lt_dev, _torch.float32)
        # cube WM latents are perview-AW: [agent(192); wrist(192)] concat (the
        # demo caches were encoded that way) — trace must match or theta* NN
        # compares 192-d against 384-d. Same shared encoder/projector per view.
        _lt_aw = (task_name == "cube")
        print(f"[latent_trace] every {_lt_every} env-steps, pool=lewm, "
              f"aw_concat={_lt_aw}, ckpt={_lt_ckpt}")

    # B-set bootstrap recording: stream this rollout into a play-format h5
    # (one episode per env, cut at first success). Toggle: ++eval.collect_h5=true.
    # See scripts/baseline/collect_rollout_h5.py for layout/alignment contract.
    _collector = None
    if bool(cfg.eval.get("collect_h5", False)):
        from collect_rollout_h5 import RolloutH5Recorder
        _col_name = str(cfg.eval.get("collect_dataset_name", None)
                        or cfg.output.run_name)
        _col_dir = str(cfg.eval.get("collect_dir", None)
                       or Path(os.environ["STABLEWM_HOME"]) / "bset")
        _collector = RolloutH5Recorder(
            Path(_col_dir) / f"{_col_name}.h5", task_name, n_eval, eval_budget,
            hold_tail=int(cfg.eval.get("collect_hold_tail", 50)))
        _sup_c = cfg.policy.get("sup", None)
        _collector_attrs = {
            "dataset_kind": "bootstrap_fixedk",
            "task": str(task_name),
            "run_name": str(cfg.output.run_name),
            "seed": int(cfg.seed),
            "fixed_k": (int(_sup_c.get("fixed_k")) if _sup_c is not None else None),
            "score_mode": (str(_sup_c.get("score_mode")) if _sup_c is not None else None),
            "hold_tail": int(cfg.eval.get("collect_hold_tail", 50)),
            "resolution": (int(cfg.world.get("resolution"))
                           if cfg.world.get("resolution") is not None else None),
            "damping": (float(cfg.world.get("damping"))
                        if cfg.world.get("damping") is not None else None),
        }
        print(f"[collect_h5] recording rollouts -> {_col_dir}/{_col_name}.h5 "
              f"(task={task_name}, n={n_eval}, budget={eval_budget})")

    for t in range(eval_budget):
        # Re-inject goal each step (legacy convention).
        world.infos.update(deepcopy(goal_step))

        # Per-step frame capture for video debug — taken BEFORE world.step() so
        # frame t shows the state the policy acts on at step st+t. This aligns
        # the agent panel's frame 0 (the reset state, step st) with the demo
        # panel's frame 0; capturing after the step would shift the agent one
        # action ahead (off-by-one vs the demo). Uses env.render() only, so it
        # is independent of world.infos (safe to read before stepping).
        # Cost: n_envs × budget × H × W × 3 bytes — watch memory on big sweeps.
        if video_frames is not None:
            cur_n_replans = _get_replan_count(policy)
            replan_flag = cur_n_replans > prev_n_replans
            prev_n_replans = cur_n_replans
            for i, env in enumerate(world.envs.unwrapped.envs):
                frame = np.asarray(env.render(), dtype=np.uint8)  # HWC uint8 RGB
                frame = _annotate_frame(frame, t, cur_n_replans, replan_flag)
                video_frames[i].append(frame)

        # [state-fork] PRE-step snapshot: mj_state[t] is the state the policy ACTS ON at
        # loop index t, i.e. the state the replan logged with env_step == t+1 planned from.
        if save_mj_state:
            mj_states.append(_capture_mj_state())

        # world.step() overwrites world.infos with the fresh env step output,
        # so behavioral capture must happen AFTER step (reads the new state).
        world.step()

        # Per-env control_timestep, read back after the step that consumed it.
        # This used to probe envs[0] only, on the stated assumption that "SAIL
        # toggles are global across the SyncVectorEnv (one majority vote per
        # step)". That assumption is what per-env AWE removes.
        _ct_req, _ct_real = _read_control_dt(world, n_eval, _default_ct)
        control_ts_per_step.append(_ct_req)
        control_ts_real_per_step.append(_ct_real)


        # [Ablation C] latent trace snapshot (post-step observation).
        if _lt_every > 0 and (t % _lt_every == 0):
            import torch as _torch

            def _lt_encode(key):
                _px = np.asarray(world.infos[key])
                if _px.ndim == 5:
                    _px = _px[:, -1]                   # (n_envs, H, W, 3) latest frame
                return _cl.encode_batch(
                    _lt_enc, _lt_proj, np.ascontiguousarray(_px, dtype=np.uint8),
                    _lt_dev, _torch.float32, _lt_mean, _lt_std, "lewm")

            _z = _lt_encode("pixels")
            if _lt_aw:
                _z = np.concatenate([_z, _lt_encode("wrist_pixels")], axis=-1)
            lt_latents.append(_z)
            lt_steps.append(t)

        # Solver step advance (MPCC needs to advance reference trajectory step
        # counter each world.step; DP/CleanDiffuser has no solver and skips).
        inner_solver = getattr(policy, "_solver", None)
        if inner_solver is not None and hasattr(inner_solver, "advance_steps"):
            inner_solver.advance_steps()

        # Capture behavioral state for metrics — but DO NOT push the dict into
        # world.infos. CEM solver iterates info_dict and tries to slice every
        # value; a nested behavioral dict would raise TypeError on slicing.
        # Behavioral goes into a separate info_for_metrics dict.
        info_for_metrics = dict(world.infos)
        if is_cube:
            beh = extract_cube_behavioral(world, n_eval, goal_pos_per_env, table_z_per_env)
            info_for_metrics.update(beh)
        elif task_name == "granular":
            from policy._runtime import extract_granular_chamfer
            info_for_metrics.update(extract_granular_chamfer(world, n_eval))
        elif task_name == "scene":
            # Two signals are published so the recipe can pick its metric:
            #   behavioral        -> SceneStageMetrics (per-stage physical
            #                        success; the meaningful one)
            #   _scene_terminated -> legacy SceneMetrics (env terminated flag).
            # The terminated flag is NOT a valid scene success signal: it
            # compares against the randomised targets drawn at reset, which the
            # scene callables never set, so even GT expert replays score 0.
            # Kept only so old scene recipes keep running unchanged.
            beh = extract_scene_behavioral(world, n_eval)
            info_for_metrics.update(beh)
            # Private key (avoids colliding with world.infos["success"]/
            # ["terminated"], which are history-stacked (n_envs, H) and would
            # make a naive per-env scalar read ambiguous).
            info_for_metrics["_scene_terminated"] = (
                np.asarray(world.terminateds, dtype=bool)
                if world.terminateds is not None
                else np.zeros(n_eval, dtype=bool)
            )
        else:  # PushT (single-view) — surface coverage + success from env state.
            info_for_metrics.update(extract_pusht_coverage(world, n_eval))
        if world.terminateds is not None:
            ever_terminated = ever_terminated | np.asarray(world.terminateds, dtype=bool)

        # Prevent gym SyncVectorEnv from auto-resetting terminated envs on the
        # next step. Matches eval_mpc_multiview.py:863 — keep terminated envs
        # frozen so behavioral metrics see the success state through end of
        # rollout.
        world.envs.unwrapped._autoreset_envs = np.zeros((n_eval,))

        # Metric update — uniform interface across all policies.
        # Read the captured executed action (stashed by the get_action wrapper
        # on world.policy). None → downstream save_traj falls back to zeros(n,2).
        action = getattr(world.policy, "_last_action_executed", None)
        if _collector is not None:
            _collector.step(world.infos, action, world.terminateds)
        metrics.update(
            obs=info_for_metrics,
            info=info_for_metrics,
            action=np.asarray(action) if action is not None else np.zeros((n_eval, 1)),
            step=t,
            env_mask=env_mask,
        )

        if save_traj and _is_scene_traj:
            _b = beh["behavioral"]
            traj["drawer_qpos"].append(np.asarray(_b["drawer_qpos"], np.float32).copy())
            traj["window_qpos"].append(np.asarray(_b["window_qpos"], np.float32).copy())
            traj["block_pos"].append(np.asarray(_b["block_pos"], np.float32).copy())
            traj["drawer_handle_y"].append(np.asarray(_b["drawer_handle_y"], np.float32).copy())
            traj["button0"].append(np.asarray(_b["button0_state"], np.int8).copy())
            traj["button1"].append(np.asarray(_b["button1_state"], np.int8).copy())
            traj["effector_pos"].append(np.asarray(_b["effector_pos"], np.float32).copy())
            traj["gripper"].append(np.asarray(_b["gripper_opening"], np.float32).copy())
            traj["action"].append(
                np.asarray(action, np.float32).copy() if action is not None
                else np.zeros((n_eval, 1), np.float32))
        elif save_traj and is_cube:
            _b = beh["behavioral"]
            traj["cube_pos"].append(np.asarray(_b["cube_pos"], np.float32).copy())
            traj["effector_pos"].append(np.asarray(_b["effector_pos"], np.float32).copy())
            traj["gripper"].append(np.asarray(_b["gripper_opening"], np.float32).copy())
            traj["action"].append(
                np.asarray(action, np.float32).copy() if action is not None
                else np.zeros((n_eval, 1), np.float32))
        elif save_traj and _is_pusht_traj:
            # world.infos fields are (n_envs, history, dims) — take the latest frame.
            def _latest(x):
                x = np.asarray(x, np.float32)
                return x[:, -1, :].copy() if x.ndim == 3 else x.copy()
            traj["block_pose"].append(_latest(world.infos["block_pose"]))   # (n,3) x,y,angle
            traj["agent_pos"].append(_latest(world.infos["pos_agent"]))     # (n,2)
            traj["vel_agent"].append(_latest(world.infos["vel_agent"]))     # (n,2)
            # block linear + angular velocity — not in info; reach into each env's
            # pymunk body (needed for an EXACT env-fork: _set_state drops these).
            _envs = world.envs.unwrapped.envs
            traj["block_vel"].append(np.array(
                [list(e.unwrapped.block.velocity) for e in _envs], np.float32))   # (n,2)
            traj["block_angvel"].append(np.array(
                [e.unwrapped.block.angular_velocity for e in _envs], np.float32))  # (n,)
            # NOTE: executed action reconstructed offline from rollout_latents.npz
            # `actions` (policy._last_action_executed is absent on GTMPCPolicy).
            traj["action"].append(
                np.asarray(action, np.float32).copy() if action is not None
                else np.zeros((n_eval, 2), np.float32))

    wall_time = time.time() - t0

    # [state-fork] terminal snapshot -> mj_states has eval_budget + 1 rows.
    if save_mj_state:
        mj_states.append(_capture_mj_state())

    if _collector is not None:
        _col_info = _collector.finalize(attrs=_collector_attrs)
        print(f"[collect_h5] wrote {_col_info['path']}: "
              f"{_col_info['episodes']} eps / {_col_info['frames']} frames "
              f"({_col_info['success_eps']} success)")

    # Phase 6: finalize + save
    results = metrics.finalize()
    results["wall_time_s"] = wall_time

    # Recompute sim-time TTC from the actual per-step dt trace, then expose
    # min/mean/max dt for traceability. For constant-dt recipes (everything
    # except SAIL-speed-adaptive) this matches the legacy ttc_sim_time_mean_s
    # exactly. For SAIL k-sweeps the actual value reflects the true wall-clock
    # speedup, not the hardcoded recipe dt × steps.
    if control_ts_per_step and "first_success_step" in results:
        ct_arr = np.asarray(control_ts_per_step, dtype=np.float64)        # (T, n_env)
        ct_real = np.asarray(control_ts_real_per_step, dtype=np.float64)  # (T, n_env)
        T = ct_arr.shape[0]
        z = np.zeros((1, ct_arr.shape[1]), dtype=np.float64)
        # Each env gets its OWN clock. The previous code built one shared 1-D
        # cumulative axis and indexed it with every env's first_success_step,
        # which is only correct while all envs run at the same rate.
        cum_req = np.concatenate([z, np.cumsum(ct_arr, axis=0)])          # (T+1, n_env)
        cum_real = np.concatenate([z, np.cumsum(ct_real, axis=0)])
        fss = np.asarray(results["first_success_step"])
        cols = np.arange(ct_arr.shape[1])
        idx_all = np.clip(fss.astype(int) + 1, 0, T)
        ttc_pe = cum_req[idx_all, cols]
        ttc_real_pe = cum_real[idx_all, cols]
        succ = fss >= 0
        if succ.any():
            results["ttc_sim_time_actual_mean_s"] = float(ttc_pe[succ].mean())
            results["ttc_sim_time_actual_std_s"] = float(ttc_pe[succ].std())
        else:
            results["ttc_sim_time_actual_mean_s"] = float("nan")
            results["ttc_sim_time_actual_std_s"] = float("nan")
        results["control_ts_min"] = float(ct_arr.min())
        results["control_ts_max"] = float(ct_arr.max())
        results["control_ts_mean"] = float(ct_arr.mean())
        results["sim_time_total_s"] = float(cum_req[-1].mean())

        # --- per-env keys (additive; the six plot readers keep using the
        # scalars above until they are migrated) -------------------------
        results["ttc_sim_time_actual_per_env_s"] = ttc_pe
        results["ttc_sim_time_realised_per_env_s"] = ttc_real_pe
        results["sim_time_total_per_env_s"] = cum_req[-1]
        results["control_ts_per_env_mean"] = ct_arr.mean(axis=0)
        # 1.0 means every env ran at the same rate on every step, i.e. the AWE
        # scheduler did nothing. This is the headline diagnostic.
        results["control_ts_n_distinct_mean"] = float(
            np.mean([len(np.unique(row)) for row in ct_arr]))
        # Fraction of (step, env) cells at the slower of the two rates —
        # compares directly against the offline precision-frame rate
        # (cube 4.3% / plain 14.2% / slippery 25.9%).
        _hi = float(ct_arr.max())
        results["control_ts_slow_frac"] = float((ct_arr >= _hi - 1e-12).mean()) if _hi > ct_arr.min() else 0.0
        results["control_ts_realised_mean"] = float(ct_real.mean())
        if abs(float(ct_real.mean()) - float(ct_arr.mean())) > 1e-9:
            print(f"[WARN] requested vs realised control_ts differ: "
                  f"{ct_arr.mean():.9f} vs {ct_real.mean():.9f} — the physics timestep "
                  f"request did not reach the solver (see _read_control_dt)")

    # Cube: emit BOTH strict GTR and loose (env-terminated-at-any-step).
    # Strict = CubeGTRMetrics' success_full (Grasped AND NoDrop AND Reached).
    # Loose = any env step reported terminated=True (matches v1 dispatcher's
    # published success_rate, which is the env-terminated criterion).
    if is_cube and "success_full" in results:
        results["success_rate_strict"] = float(np.asarray(results["success_full"]).mean())
        results["success_rate_loose"] = float(ever_terminated.mean())

    outdir = _output_dir(cfg)
    _save_results(results, cfg, outdir)

    # Full dt trace + the raw precision bits. ~700 KB at (1782, 50) float64.
    # Worth keeping unconditionally because it separates two failure modes that
    # results.json cannot: "the scheduler never diverged" (control_ts uniform)
    # from "the DP's precision head never fired" (precision_bits all False).
    # The smoke measured a slow-step fraction of 12.1% against a 25.9% offline
    # label rate on slippery, and only this trace can say which end that gap is.
    if control_ts_per_step:
        _tr = {"requested": np.asarray(control_ts_per_step, dtype=np.float64),
               "realised": np.asarray(control_ts_real_per_step, dtype=np.float64),
               "first_success_step": np.asarray(results.get("first_success_step", [])),
               "per_env_success": np.asarray(results.get("per_env_success",
                                                         results.get("success_full", []))),
               "eval_budget": np.asarray(eval_budget)}
        _pb = getattr(policy, "precision_bits_trace", None)
        if _pb is not None:
            _tr["precision_bits"] = np.asarray(_pb)
        np.savez_compressed(outdir / "control_ts_trace.npz", **_tr)

    # [dist-shift ablation] dump per-solve deployment latents (cur_emb) for the
    # WM↔deployment observation-gap (sliced-W1). Accumulated in MPCCSolver.solve
    # (latent_modules/mpcc/solver.py); MPCC-only (DP/other policies lack _solver
    # or _cur_emb_history → skipped). File is small: (n_solves, n_envs, D).
    _solver = getattr(policy, "_solver", None)
    _ceh = getattr(_solver, "_cur_emb_history", None) if _solver is not None else None
    if _ceh:
        _lat = np.stack(_ceh, axis=0)  # (n_solves, n_envs, D)
        _save_kw = dict(
            latents=_lat,
            first_success_step=np.asarray(results.get("first_success_step", []), dtype=np.float64),
            eval_budget=int(eval_budget),
        )
        # [density-proxy A2] chosen action plan per solve (n_solves, n_envs, H, A)
        # → enables (z, a) proxy: z=latents[k,e], a=actions[k,e,0] (first block).
        _slog = getattr(_solver, "_solve_log", None)
        if _slog:
            try:
                _save_kw["actions"] = np.stack(
                    [np.asarray(d["actions"].cpu().numpy() if hasattr(d["actions"], "cpu")
                                else d["actions"]) for d in _slog], axis=0)
            except Exception as _ae:
                print(f"[density-proxy] action logging skipped: {_ae!r}")
        np.savez_compressed(outdir / "rollout_latents.npz", **_save_kw)
        print(f"[eval_inprocess] rollout_latents.npz saved: latents={_lat.shape} "
              f"+ first_success_step + eval_budget={eval_budget}")

    # [Ablation C] policy-independent latent trace dump (any policy).
    if _lt_every > 0 and lt_latents:
        _ltz = np.stack(lt_latents, axis=0)            # (n_snap, n_envs, D)
        np.savez_compressed(
            outdir / "latent_trace.npz", latents=_ltz,
            steps=np.asarray(lt_steps, dtype=np.int64),
            every=np.int64(_lt_every),
            first_success_step=np.asarray(
                results.get("first_success_step", []), dtype=np.float64),
        )
        print(f"[eval_inprocess] latent_trace.npz saved: {_ltz.shape} "
              f"(every {_lt_every} env-steps)")

    # [Ablation C] perturbation record: base + perturbed init states per env,
    # for radius-vs-outcome analysis and exact reproduction of the init set.
    if init_perturb_fn is not None and _perturb_rec:
        np.savez_compressed(
            outdir / "init_perturb.npz",
            base_state=_perturb_rec["base"], perturbed_state=_perturb_rec["perturbed"],
            xy_std=np.float64(_p_blk if _p_blk > 0 else _p_xy),
            theta_std=np.float64(_p_th),
            seed=np.int64(_p_seed))
        print(f"[eval_inprocess] init_perturb.npz saved "
              f"(n={_perturb_rec['base'].shape[0]})")

    # [state-fork] MuJoCo state trace. mj_state[t] is the PRE-step state at loop index t,
    # so mj_state[0] is post-reset and mj_state[eval_budget] is terminal. A replan logged in
    # cand_arrays.npz with env_step == s was planned from mj_state[s - 1].
    if save_mj_state and mj_states:
        _mjz = np.stack(mj_states, 0)
        np.savez_compressed(
            outdir / "rollout_mj_state.npz",
            mj_state=_mjz,
            state_sig=np.int64(_mj_sig_int),
            state_size=np.int64(_mj_size),
            index_convention="pre_step: mj_state[t] = state BEFORE world.step() #t",
            episode_idx=np.asarray(ns.eval_episodes, dtype=np.int64),
            start_step=np.asarray(ns.eval_start_idx, dtype=np.int64),
            eval_budget=int(eval_budget),
        )
        print(f"[eval_inprocess] rollout_mj_state.npz saved: {_mjz.shape} "
              f"(T+1, n_envs, S={_mj_size})")

    if save_traj and traj is not None and _is_scene_traj and traj["drawer_qpos"]:
        _sfs = results.get("stage_first_step", {}) or {}
        np.savez_compressed(
            outdir / "rollout_traj.npz",
            drawer_qpos=np.stack(traj["drawer_qpos"], 0),          # (T, n_envs)
            window_qpos=np.stack(traj["window_qpos"], 0),          # (T, n_envs)
            block_pos=np.stack(traj["block_pos"], 0),              # (T, n_envs, 3)
            drawer_handle_y=np.stack(traj["drawer_handle_y"], 0),  # (T, n_envs)
            button0=np.stack(traj["button0"], 0),                  # (T, n_envs)
            button1=np.stack(traj["button1"], 0),                  # (T, n_envs)
            effector_pos=np.stack(traj["effector_pos"], 0),        # (T, n_envs, 3)
            gripper=np.stack(traj["gripper"], 0),                  # (T, n_envs)
            action=np.stack(traj["action"], 0),                    # (T, n_envs, A)
            segment=str(results.get("segment", "")),
            first_success_step=np.asarray(results.get("first_success_step", []), dtype=np.float64),
            **{f"stage_first_step_{k}": np.asarray(v, dtype=np.float64)
               for k, v in _sfs.items()},
            episode_idx=np.asarray(ns.eval_episodes, dtype=np.int64),
            start_step=np.asarray(ns.eval_start_idx, dtype=np.int64),
            eval_budget=int(eval_budget),
        )
        print(f"[eval_inprocess] rollout_traj.npz saved: scene drawer/window/block/"
              f"button/eff/grip/action ({len(traj['drawer_qpos'])} steps × {n_eval} envs)")
    elif save_traj and traj is not None and is_cube and traj["cube_pos"]:
        np.savez_compressed(
            outdir / "rollout_traj.npz",
            cube_pos=np.stack(traj["cube_pos"], 0),          # (T, n_envs, 3)
            effector_pos=np.stack(traj["effector_pos"], 0),  # (T, n_envs, 3)
            gripper=np.stack(traj["gripper"], 0),            # (T, n_envs)
            action=np.stack(traj["action"], 0),              # (T, n_envs, A)
            goal_pos=np.asarray(goal_pos_per_env, dtype=np.float32),
            table_z=np.asarray(table_z_per_env, dtype=np.float32),
            first_success_step=np.asarray(results.get("first_success_step", []), dtype=np.float64),
            success_full=np.asarray(results.get("success_full", []), dtype=bool),
            episode_idx=np.asarray(ns.eval_episodes, dtype=np.int64),
            start_step=np.asarray(ns.eval_start_idx, dtype=np.int64),
            eval_budget=int(eval_budget),
        )
        print(f"[eval_inprocess] rollout_traj.npz saved: cube/eff/grip/action "
              f"({len(traj['cube_pos'])} steps × {n_eval} envs) + episode_idx + start_step")
    elif save_traj and traj is not None and _is_pusht_traj and traj["block_pose"]:
        from policy._runtime import PUSHT_FIXED_GOAL as _PFG
        np.savez_compressed(
            outdir / "rollout_traj.npz",
            block_pose=np.stack(traj["block_pose"], 0),      # (T, n_envs, 3) x,y,angle
            agent_pos=np.stack(traj["agent_pos"], 0),        # (T, n_envs, 2)
            vel_agent=np.stack(traj["vel_agent"], 0),        # (T, n_envs, 2)
            block_vel=np.stack(traj["block_vel"], 0),        # (T, n_envs, 2)
            block_angvel=np.stack(traj["block_angvel"], 0),  # (T, n_envs)
            action=np.stack(traj["action"], 0),              # (T, n_envs, 2) executed raw
            goal_state=np.asarray(_PFG, dtype=np.float64),
            first_success_step=np.asarray(results.get("first_success_step", []), dtype=np.float64),
            success_full=np.asarray(results.get("success_full", []), dtype=bool),
            episode_idx=np.asarray(ns.eval_episodes, dtype=np.int64),
            start_step=np.asarray(ns.eval_start_idx, dtype=np.int64),
            eval_budget=int(eval_budget),
        )
        print(f"[eval_inprocess] rollout_traj.npz (pusht) saved: block/agent/vel/action "
              f"({len(traj['block_pose'])} steps × {n_eval} envs)")

    # Write per-env videos if captured.
    if video_frames is not None:
        import imageio
        video_dir = outdir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        # Play back in SIM TIME, not one-frame-per-control-step. Under SAIL a
        # control step covers `ct` seconds of sim time, and `ct` differs between
        # the fast and slow branches — so a uniform fps renders both branches at
        # the same apparent speed and hides the only thing the video is for.
        # Emit at the finest period present and hold each frame for ct/unit
        # frames, which makes the slow branch visibly linger.
        _ct = (np.asarray(control_ts_per_step, dtype=np.float64)
               if control_ts_per_step else None)                 # (T, n_env)
        _unit = (float(np.nanmin(_ct)) if _ct is not None and _ct.size
                 else float(results.get("control_timestep_s", 0.1)))
        fps = max(1, int(round(1.0 / _unit)))
        _up = int(cfg.eval.get("video_upscale", 1) or 1)
        for i, frames in enumerate(video_frames):
            if not frames:
                continue
            if _ct is not None and _ct.shape[0] >= len(frames):
                # Annotate BEFORE duplicating, so the label costs one draw per
                # control step rather than one per emitted frame.
                frames = [_annotate_rate(f, float(_ct[t, i]), _default_ct, _up)
                          for t, f in enumerate(frames)]
                reps = np.maximum(1, np.round(_ct[:len(frames), i] / _unit)).astype(int)
                frames = [f for f, r in zip(frames, reps) for _ in range(int(r))]
            path = video_dir / f"env_{i:02d}.mp4"
            imageio.mimsave(str(path), frames, fps=fps)
        print(f"[eval_inprocess] saved {len(video_frames)} videos to {video_dir}/")

        # [Option B] Reference-vs-agent comparison. Render the GT demo each env
        # tracked (the eval-dataset chunk it was reset to + reference) and stack
        # it beside the agent rollout → ref_env_i.mp4 + compare_env_i.mp4
        # (reference | agent). Lets you see whether the policy follows the demo's
        # PATH (just faster), not the agent in isolation. ns.eval_episodes /
        # eval_start_idx / dataset are all in scope from setup + Phase 3.
        ep_ids = np.asarray(ns.eval_episodes).tolist()
        starts = np.asarray(ns.eval_start_idx).tolist()
        n_ref = 0
        for i, frames in enumerate(video_frames):
            if not frames:
                continue
            try:
                ep, st = int(ep_ids[i]), int(starts[i])
                chunk = list(dataset.load_chunk([ep], [st], [st + int(eval_budget)]))[0]
                px = chunk["pixels"]
                px = px.permute(0, 2, 3, 1) if hasattr(px, "permute") else px
                ref = np.asarray(px, dtype=np.uint8)
                if ref.ndim != 4 or ref.shape[0] == 0:
                    continue
                imageio.mimsave(str(video_dir / f"ref_env_{i:02d}.mp4"), list(ref), fps=fps)
                ag = np.asarray(frames, dtype=np.uint8)
                if ag.shape[1:] == ref.shape[1:]:  # same H,W,C → render side-by-side
                    T = max(len(ref), len(ag))
                    refp = ref if len(ref) >= T else np.concatenate(
                        [ref, np.repeat(ref[-1:], T - len(ref), 0)], 0)
                    agp = ag if len(ag) >= T else np.concatenate(
                        [ag, np.repeat(ag[-1:], T - len(ag), 0)], 0)
                    # demo | agent side-by-side.
                    comp = np.concatenate([refp, agp], axis=2)        # (T, H, 2W, 3)
                    imageio.mimsave(str(video_dir / f"compare_env_{i:02d}.mp4"), list(comp), fps=fps)
                n_ref += 1
            except Exception as exc:
                print(f"[eval_inprocess] WARN: ref video env {i} failed: {exc!r}")
        if n_ref:
            print(f"[eval_inprocess] saved {n_ref} ref_env_*.mp4 (+ compare_env_*.mp4) "
                  f"[reference|agent] to {video_dir}/")

    sr = float(results.get("success_rate", float("nan")))
    print("=" * 80)
    print(f"Success rate: {sr*100:.1f}%")
    if "ttc_steps_mean" in results and results["ttc_steps_mean"] == results["ttc_steps_mean"]:
        print(f"TTC: {float(results['ttc_steps_mean']):.1f} steps → "
              f"{float(results.get('ttc_sim_time_mean_s', float('nan'))):.2f}s sim-time")
    print(f"Saved to: {outdir}/")
    print(f"Wall: {wall_time:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
