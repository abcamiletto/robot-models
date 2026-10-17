# Visual and collision meshes

Every canonical hand supplies both visual and collision geometry. The same hand,
joints and FK drive both representations:

```python
from robot_models import create_model

hand = create_model("shadow", side="right")
params = hand.get_rest_pose(hands="rest")
visual = hand.forward_meshes(**params)
collision = hand.forward_meshes(geometry="collision", **params)
```

`link_meshes` contains the existing visual meshes. `collision_meshes` contains
one mesh per source collision geom, in its skeleton joint's local Y-up frame.
`collision_names` and `collision_joint_indices` identify those pieces. Their
world transforms are returned by `forward_links(geometry="collision", **params)`.
A joint can carry several collision pieces, so their count differs from the
visual mesh count. NumPy, PyTorch and JAX share this interface; array-valued
transforms remain differentiable, while `Trimesh` outputs use host NumPy arrays.

## Source choices

We favor widely used, maintained descriptions that match the canonical physical
hand: MuJoCo Menagerie, manufacturer models, and established ROS integrations.
The source definitions determine the geometry; visual meshes are never
silently replaced by automatically generated collision hulls.

| Hand | Collision source | Visual pieces | Collision pieces | Native collision shapes |
| --- | --- | ---: | ---: | --- |
| Ability | PSYONIC API | 12 | 12 | 11 meshes, 1 cylinder |
| Allegro | Menagerie | 21 | 21 | 17 boxes, 4 capsules |
| Barrett BH280 | Robotnik URDF | 10 | 10 | 9 meshes, 1 box |
| BrainCo Revo 2 | Manufacturer SDK | 22 right / 19 left | 22 right / 17 left | meshes |
| EZGripper | SAKE-maintained MJCF | 9 | 19 | 7 meshes, 12 boxes |
| Inspire | Unitree integration | 13 | 13 | meshes |
| LEAP | Menagerie | 17 | 71 | 67 boxes, 4 meshes |
| Robotiq 3-Finger | Robotiq-authored ROS URDF | 13 | 13 | dedicated collision meshes |
| Sharpa Wave | Manufacturer MJCF | 28 | 26 | meshes |
| Shadow E3M5 | Menagerie | 24 | 38 | 12 boxes, 10 capsules, 7 cylinders, 6 meshes, 3 spheres |

Counts apply to both supplied sides unless specified. Source authors sometimes
omit a collider for a visual joint, such as Sharpa's two thumb linkage pieces;
we retain their authored coverage. See [hand sources](hand-sources.md) for exact
upstream revisions, qualifications and license terms.

[Menagerie's Shadow](https://github.com/google-deepmind/mujoco_menagerie/blob/f054586a8e90465d49ee5be15335c4a0c7f57caf/shadow_hand/right_hand.xml),
[Allegro](https://github.com/google-deepmind/mujoco_menagerie/blob/f054586a8e90465d49ee5be15335c4a0c7f57caf/wonik_allegro/right_hand.xml)
and [LEAP](https://github.com/google-deepmind/mujoco_menagerie/blob/f054586a8e90465d49ee5be15335c4a0c7f57caf/leap_hand/right_hand.xml)
provide the widely used simulation definitions for those hands.
[DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet/blob/bd1b13d7248729af117e1d46aaa6266b147a3c7b/grasp_generation/utils/hand_model.py)
uses analytic capsule distances and box/mesh surfaces for its Shadow optimizer;
[GenDexGrasp](https://github.com/tengyu-liu/GenDexGrasp/blob/29fe7efc558b8cc3ce2d8e67c9c22b014adc21c7/utils_model/HandModel.py)
loads visual surfaces for contact-map optimization. We use these as comparison
references while preserving each canonical source's collision definitions and
joint/root conventions. No extra hand variants are introduced.

## Meshes and native physics

The Python mesh view triangulates spheres, capsules, cylinders and ellipsoids;
curved surfaces therefore have a finite tessellation. Boxes are exact. Authored
collision meshes retain their source triangles and transforms.

`add_hand(scene, name)` attaches the original MJCF collision geoms. Native
primitives stay analytic, and collision masks, explicit pairs and exclusions
continue to be enforced by MuJoCo. MuJoCo uses the convex hull of an authored
mesh geom for collision detection; the Python mesh interface returns its original
triangles. These behaviors are documented in the
[MuJoCo geom/mesh reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom).

Barrett and Robotiq retain all original URDF collision elements. Robotiq uses its
separate simplified collision STLs, including their nonidentity origins. Added
collision geoms have zero density so that restoring them preserves existing
runtime masses. Those models still use placeholder inertias and require
application-appropriate dynamics/actuation for simulation; URDF contact
coefficients are retained in source files, rather than translated into solver
parameters.

Shadow restores Menagerie's forearm collision mesh on both sides, which the
previous package omitted. All existing visual mesh bytes, skeleton frames,
joint conventions and other geoms are retained.

## Updating existing assets

Managed caches refresh automatically for the updated Barrett, Robotiq and Shadow
bundles. Explicit/configured asset directories remain under the caller's control;
refresh an older first-party bundle with:

```bash
robot-models download shadow --output-dir /path/to/shadow
```

## Validation and rebuilding

The shared model tests compare collision pieces with MuJoCo at articulated poses,
including source primitive dimensions, mesh support surfaces and joint frames.
An independent MuJoCo distance-query test covers inherited defaults, explicit
contact pairs and alternative primitive orientations. Backend consistency tests
exercise visual and collision meshes at nonzero poses across NumPy/Torch/JAX.

The original-source fixture generator also validates Barrett, Robotiq and
EZGripper collision surfaces at every fixed reference pose. Maximum surface
conversion error is below 3e-9 m. Rebuild these bundles with the commands in
[hand sources](hand-sources.md).

For Shadow, use the pinned Menagerie checkout and the original
[canonical archive](https://huggingface.co/abcamiletto/robot-models/resolve/2fd448dce9a30b38cf4f106086bd7e2f13781f03/shadow/assets.zip):

```bash
uv run --group asset-preparation python -m scripts.prepare_shadow_assets \
  UPSTREAMS/menagerie ORIGINAL_SHADOW_ZIP results/shadow-collisions
uv run --group asset-preparation python -m scripts.check_shadow_fidelity \
  UPSTREAMS/menagerie results/shadow-collisions/assets results/shadow-check.json
```

The latter imports untouched upstream MJCF files separately from the runtime
bundle and compares both sides at five poses. Body-transform error is zero;
original visual and collision mesh support surfaces agree within 8e-9 m.
