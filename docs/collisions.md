# Visual and collision meshes

All hands and humanoids provide visual and collision meshes through the same API:

```python
from robot_models.hands.shadow.numpy import ShadowHand

hand = ShadowHand()
params = hand.get_rest_pose()
visual = hand.forward_meshes(**params)
collision = hand.forward_meshes(geometry="collision", **params)
```

`link_meshes` and `collision_meshes` return local meshes in each joint's Y-up
frame. `collision_names` and `collision_joint_indices` identify collision pieces;
`forward_links(geometry="collision", **params)` returns their world transforms.
A joint can have several pieces. Transforms use the chosen array backend;
`Trimesh` outputs use host NumPy arrays.

## Source geometry

Collisions follow the [canonical sources](hand-sources.md):

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


Counts apply to both sides unless stated. Coverage follows the source; for
example, Sharpa omits colliders on two thumb linkage pieces.

| Humanoid | Description | Collision pieces |
| --- | --- | ---: |
| G1 | LeRobot / Unitree | 39 |
| GR1 | Fourier | 15 |
| H1 | Menagerie / Unitree | 32 |
| T1 | Menagerie / Booster | 12 |
| TALOS | PAL Robotics | 7 |


## Native physics

The Python mesh API preserves source mesh triangles and triangulates curved
primitives. `add_hand` keeps native analytic shapes, collision masks, contact
pairs and exclusions. MuJoCo uses a convex hull for each mesh geom's collision
detection; see its [geom reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom).

Barrett and Robotiq retain their URDF collision elements, including Robotiq's
separate collision STLs and offsets. Added collision geoms have zero density to
preserve existing masses. Their inertias are placeholders; source URDF contact
coefficients are retained but are not converted into MuJoCo solver settings.

Converted URDFs exclude adjacent links from self-collision. Robotiq also excludes
the palm against each first finger link because those meshes always overlap.
Object contacts and other self-collisions remain enabled. `source.json` records
these changes. Shadow includes Menagerie's forearm collider on both sides.

## Asset updates and validation

Managed caches update automatically. Older explicit or configured bundles are
rejected and left intact. Refresh one with:

```bash
robot-models download shadow --output-dir /path/to/shadow
```

Tests check joint frames, primitive dimensions and mesh surfaces against MuJoCo
at articulated poses, plus consistency across NumPy, Torch and JAX. Independent
upstream comparisons have surface error below 3e-9 m for Barrett, Robotiq and
EZGripper, and below 8e-9 m for Shadow. See [validation records](hand-sources.md).
