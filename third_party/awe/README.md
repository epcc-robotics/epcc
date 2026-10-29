# AWE waypoint extraction (vendored subset)

`waypoint_extraction/` is copied from AWE (Automatic Waypoint Extraction),
https://github.com/lucys0/awe, commit `9780e41` (2023-11-12). Only this
package is vendored; the rest of AWE (robomimic/robosuite/ACT/diffusion
policy integrations, media) is not used here.

It is used by `scripts/sail/make_labels.py` and `scripts/sail/make_labels_pusht.py`
to compute the SAIL precision labels (`dp_waypoint_selection` with `env=None`).
The label scripts put this directory on `sys.path` (`AWE_ROOT`) and install a
scipy-based stand-in for `robosuite.utils.transform_utils.quat_slerp` before
importing, so robosuite is not required.

## Patches (both applied)

- `patches/01_point_line_distance_degenerate.patch`: a zero-length segment
  (two consecutive waypoints at the same position) made the point-to-segment
  distance 0/0 = NaN, which poisoned the reconstruction error and prevented
  the dynamic-programming selection from ever accepting a stationary stretch.
  Fixed to use the point distance. All SAIL labels reported were produced
  with this fix (research-machine patch, 2026-08-01).
- `patches/02_lazy_unused_imports.patch`: moves the `wandb` and AWE `utils`
  imports (used only by the simulator-based trajectory replay and video
  logging) inside the functions that use them. No effect on
  `dp_waypoint_selection`; it only removes hidden dependencies.

## License

Upstream `setup.py` declares `license="MIT License"`, but the upstream
repository does not ship a LICENSE file, so no license text is reproduced
here. Copyright remains with the AWE authors (Lucy Xiaoyang Shi et al.,
"Waypoint-Based Imitation Learning for Robotic Manipulation", CoRL 2023).
