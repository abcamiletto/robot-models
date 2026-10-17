# Architecture

Robots are data definitions. Model-specific NumPy, Torch and JAX imports bind a
canonical definition to its shared family implementation, following Body Models:

```python
from robot_models.hands.shadow.torch import ShadowHand

hand = ShadowHand(side="right")
collision = hand.forward_meshes(geometry="collision", **hand.get_rest_pose())
```

| Module | Responsibility |
| --- | --- |
| `hands/_models.py`, `humanoids/_models.py` | Canonical identifiers, anatomical mappings, supplied sides and asset requirements. |
| `_spec.py`, `_catalog.py` | Immutable definition schema and combined catalog. |
| `hands/_base.py`, `humanoids/_base.py` | One implementation per family: construction, pose parameters and presets. |
| `<family>/<model>/numpy.py`, `torch.py`, `jax.py` | Declare the model-specific import; `_backend.py` binds its data to the shared family and runtime. |
| `_base.py` | Shared model contract and immutable identity. |
| `_assets.py` | Resolve asset directories and download complete bundles from Hugging Face. |
| `_common/mjcf.py` | Read source MJCF into skeleton, hinge, coupling and geometry data. |
| `_common/rigid.py` | Shared kinematics and mesh assembly. |

Each definition selects one canonical description. The loader reads skeleton
joints as MJCF bodies in body order; independent hinges are pose coordinates in
MuJoCo `qpos` order. A hinge named as `joint1` of an equality follows
`joint1 = polycoef(joint2)`. All hinges therefore use the same polynomial
coupling path, followed by the same FK. Outputs use the Y-up model basis, with
the root at the origin.

Loaded state separates skeleton/hinge data from equivalent `visual` and
`collision` mesh groups. Each group packs vertices and faces once. Offset arrays
delimit the pieces and attach each to a skeleton joint. Posing transforms those
ranges, joins the arrays once and creates one independent `Trimesh` per batch
element. Local mesh properties return independent snapshots of current state.
Both families expose local `collision_meshes` and posed
`forward_meshes(geometry="collision", **params)`.

Torch materializes both groups as module buffers; JAX registers the same data
as a pytree. The immutable model configuration carries its canonical name and
hand side through serialization and JAX transformations. Array-valued transforms
remain native to the selected runtime until a `Trimesh` output needs host arrays.

## Adding a model

1. Upload `<name>/assets.zip` and `<name>/LICENSE.md` to the Hugging Face repository.
   The archive holds `<name>.xml` for a humanoid, or the supplied hand's
   `left.xml`/`right.xml`, meshes, original descriptions and license notices.
2. Add a `ModelSpec` entry to its family's `_models.py` and package its presets as
   `<family>/<model>/assets/poses.npz` (`t_pose`/`a_pose` or `{side}_flat`/`{side}_rest`).
3. Add a model documentation page and validate its definition against the source.

Add three backend declarations with `model_for_backend(name, backend, module=__name__)`.
Loading, joint couplings, FK and mesh assembly remain in the shared implementation.
Source collision primitives remain analytic in MuJoCo; the mesh interface
triangulates curved shapes. Canonical sources and preparation are recorded in
[hand sources](hand-sources.md) and [collision geometry](collisions.md).
