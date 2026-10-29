# third_party

Exact versions of the external code that produced the paper's numbers.

| Directory | What | Source |
|---|---|---|
| `stable_worldmodel/` | PushT and Cube environments, wrappers, CEM solver, data utilities | submodule, [epcc-robotics/stable-worldmodel-srl](https://github.com/epcc-robotics/stable-worldmodel-srl) branch `epcc-release` (MIT) |
| `cleandiffuser/` | diffusion backbones of the contour generator and the DP / SkiP / SAIL baselines | submodule, [epcc-robotics/CleanDiffuser](https://github.com/epcc-robotics/CleanDiffuser) branch `epcc-release` (Apache-2.0) |
| `awe/` | AWE waypoint selection for the SAIL precision labels | vendored subset (see `awe/README.md`) |

Clone with `git clone --recursive`, or run `git submodule update --init` in an existing clone.
Both submodules are installed into the environment without dependencies (the top-level
requirements cover them):

```bash
pip install --no-deps -e third_party/stable_worldmodel -e third_party/cleandiffuser
```

## stable-worldmodel

The lab fork of [galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
at commit `5aeabb3` (merge of the fork's `main` `68824a4` and branch `escapable-pusht` `0a0f530`,
upstream package version 0.0.6). Relative to upstream it adds PushT damping, an escapable block, and
exact substep rounding for the speed-up wrappers. These are the environment semantics behind all
reported numbers, so this version, not the PyPI release, must be used to reproduce them.

The `epcc-release` branch adds two commits on top of `5aeabb3`: the upstream MIT `LICENSE` file
(missing in the lab fork) and a packaging change (build backend uv_build to setuptools,
optional-dependency groups dropped, version `0.0.6+srl`). There are no source changes.

`patches/stable_worldmodel_pusht_block_vel_obs.patch` is an uncommitted change from the research
machine that adds a `block_vel` observation key. It is not applied: the evaluation harness reads block
velocity directly from the physics body and nothing in the reproduction chain consumes that key. It is
kept only so the research environment can be reconstructed exactly.

## CleanDiffuser

[CleanDiffuserTeam/CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) at commit
`05f17fc` (2025-04-20). The `epcc-release` branch adds two commits:

- `cleandiffuser/nn_diffusion/chiunet.py`: upstream `ChiUNet1d` asserts that the action horizon is a
  power of two. The contour generator uses a horizon of 84, so the assert is relaxed to "divisible by
  2^(number of down-sampling levels)", the U-Net's actual requirement. Models with power-of-two
  horizons (the DP / SkiP / SAIL baselines) are unaffected.
- Packaging: upstream pins `numba<0.60`, which forces `numpy<2` and conflicts with this environment.
  The release build declares no dependencies and has the static version `0.1.0+h84`.
