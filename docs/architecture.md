# Architecture

Each robot is a data definition bound to one shared family implementation.
The import selects NumPy, Torch or JAX.

| Module | Purpose |
| --- | --- |
| `hands/_models.py`, `humanoids/_models.py` | Names, anatomical mappings, sides, symmetry rules and asset versions |
| `_spec.py`, `_catalog.py` | Definition schema and model catalog |
| `hands/_base.py`, `humanoids/_base.py` | Family constructors, poses and presets |
| `<family>/<model>/{numpy,torch,jax}.py` | Public classes, bound by `_backend.py` |
| `_base.py` | Shared model API and identity |
| `_assets.py` | Asset paths and Hugging Face downloads |
| `_common/mjcf.py` | MJCF parsing |
| `_common/rigid.py` | Kinematics and mesh assembly |

Skeleton joints follow MJCF body order. Pose coordinates follow independent
hinges in MuJoCo `qpos` order. Coupled hinges use `joint1 = polycoef(joint2)`
before FK. Outputs use a Y-up frame with the root at the origin.

Visual and collision groups store packed vertices, faces and link indices.
Posing creates one independent `Trimesh` per batch element; local mesh properties
return copies. Torch stores arrays as module buffers. JAX uses pytrees with the
model name and side as static metadata. Arrays stay in their backend until mesh
output converts them to host NumPy arrays.

## Add a model

1. Publish `<name>/assets.zip` and `<name>/LICENSE.md` on Hugging Face. Include
   `<name>.xml` for a humanoid or `left.xml`/`right.xml` for the supplied hand sides,
   meshes, original descriptions and licenses.
2. Add a `ModelSpec` to the family's `_models.py` and presets in
   `<family>/<model>/assets/poses.npz`: `t_pose`/`a_pose` or `{side}_flat`/`{side}_rest`.
3. Add three backend modules using `model_for_backend(name, backend, module=__name__)`.
4. Document the model and check its FK, joints and geometry against the source.

Keep loading, couplings and FK in the shared implementation. See
[hand sources](hand-sources.md) and [collision geometry](collisions.md).
