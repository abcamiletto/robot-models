# Baseline hand fidelity record

Checked on 2026-10-17 against the original pinned implementations, using this
checkout's `src/robot_models` and freshly prepared source bundles. No baseline
source or frozen grasp prediction store was edited. Both vendored kinematics
packages ran in separate processes.

References: GenDexGrasp `29fe7efc558b8cc3ce2d8e67c9c22b014adc21c7`,
DexGraspNet `bd1b13d7248729af117e1d46aaa6266b147a3c7b`, and the original
GenDexGrasp archive SHA-256
`c8777d2195da312281f51c37cdadd8a46bb1b408aada2e9ffd425bd237357f4e`.
See [preparation and reproduction commands](baselines.md).

The final audit also runs in the declared `baseline-audit` uv dependency group:
Python 3.12, NumPy 2.4.1, PyTorch 2.10.0, Trimesh 4.11.1. MuJoCo checks use 3.10.0.
The supplied older audit interpreter (PyTorch 2.7.1) was independently used during
implementation with the same numerical results. Checks here run on CPU.

| Variant | Fixed poses | Largest FK matrix error | Largest mesh vertex error (m) | Largest annotation position error (m) |
| --- | ---: | ---: | ---: | ---: |
| `barrett-gendexgrasp` | 14 | 1.79e-7 | 3.71e-9 | 2.96e-8 |
| `robotiq-3finger-gendexgrasp` | 17 | 1.79e-7 | 0 | 1.89e-8 |
| `ezgripper-gendexgrasp` | 10 | 2.38e-7 | 6.59e-9 | 5.84e-9 |
| `shadow`, `source="gendexgrasp"` | 30 | 2.98e-7 | 8.19e-9 | 4.63e-8 |
| `shadow`, `source="dexgraspnet"` | 28 | 1.79e-7 | 7.68e-9 (boxes) | 1.67e-8 |

The pose set contains zero, clipped zero, both joint-limit endpoints, two interior
poses, and one independent perturbation per coordinate. Joint names, ordering and
limits are checked against the real source kinematics/URDFs, not a second
robot-models model. Full rotations and translations, including rest transforms,
are compared at every link. Every original visual mesh vertex/triangle is checked
for the GenDexGrasp hands, followed by representative triangle interior points
through articulated FK. GenDexGrasp annotation errors cover the contact bases
resolved by the actual original `HandModel`.

DexGraspNet checks all four box mesh links against the original bevelled `box.obj`,
and every vertex on all 15 capsule meshes against native analytic capsules. The
largest capsule surface residual is **1.77e-9 m**. All **140** palmar contacts and
**19** self-penetration annotations are transformed through their exact original
moving `*_child` frames at every pose. The extra fixed links of the vendored MJCF
parser are preserved. Native collision definitions and link exclusions remain
available to the original hand algorithm; this test does not replace its SDF
implementation with runtime mesh hulls.

Existing left Allegro was checked over 22 fixed poses using all 16 articulated
bodies and explicit per-body constant rest-frame mappings. Its largest FK error
is **1.18e-6**, with identical mechanical signs and joint limits within **5.35e-8 rad**.
For 30,000 surface samples at each of rest and two articulated poses, mean
original-to-canonical sample distance is **1.017–1.018 mm**; the original-to-original
independent-sampling control is **1.015–1.017 mm**. The 95th percentiles are
**1.987–1.995 mm**, versus **1.984–1.987 mm** for the control. This supports reusing
`allegro(side="left")`; no extra Allegro runtime model was added. Full joint/rest
matrices and surface diagnostics are emitted to `results/fidelity/allegro-left.json`.

The implementation has synthetic regressions for fixed wrist rotation, negative
axes, independent mimic coordinates, Shadow's mesh basis adjustment, bevelled
boxes, capsule centering, native annotation paths, source tamper detection and
unpublished-asset errors. Independent upstream fixtures run against NumPy, Torch
and JAX. The existing cross-model checks additionally cover MuJoCo body/visual
agreement, batching, pose helpers, backend consistency, gradients, Torch compilation,
JAX JIT and JAX state round trips for all five new variants.

The DexGraspNet source bundle is published on the asset hub with its original
licenses. GenDexGrasp archives and source-derived numerical fixtures remain local
under `results/` while their redistribution terms are unresolved. Ordinary CI can
run the synthetic regressions without them.
Missing local bundles/fixtures cause explicitly explained skips in optional
integration tests; the independent audit CLI fails rather than skipping.

## Published DexGraspNet bundle

Hugging Face snapshot
[`3788bdcfc8082a3c9bc696da5bbff8f8fd90ebaf`](https://huggingface.co/abcamiletto/robot-models/commit/3788bdcfc8082a3c9bc696da5bbff8f8fd90ebaf)
contains `shadow-dexgraspnet/assets.zip` and `LICENSE.md` with the per-folder
license table. The review update changes only the bundle notice and manifest: it
narrows the Apache scope to the code identified in the unchanged inherited notice
and records license-file hashes. Source files and runtime geometry are identical
to the first publication; every pre-existing model folder is unchanged.

Archive SHA-256:
`b0afd0e9fd9a5144f2151942c99fb89135e9334c6371e6452e29e2dd3265c159`.

Anonymous download, archive digest, all 26 original-file hashes, both license-file hashes, license contents
and runtime loading were verified after publication. The downloaded bundle was
also checked against the pinned DexGraspNet implementation over the same 28 poses,
including joint order/limits, mesh surfaces and all native annotations.

These measurements establish source-hand fidelity at the checked poses. They do
not claim original dataset grasp success rates or Isaac Gym stability reproduction.
