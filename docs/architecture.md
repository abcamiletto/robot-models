# Architecture

Models use MuJoCo MJCF files read by one loader. Public assets are hosted on Hugging Face. Each hand has one canonical upstream description:

Model packages live under `robot_models.hands` and `robot_models.humanoids`.
Both families use the root `create_model` factory and the same visual/collision
mesh API. Shared code stays at the package root; no robot implementation is duplicated.

| Module | Responsibility |
| --- | --- |
| `_assets.py` | Resolve a model's asset directory: explicit path, config, cache, or download. |
| `_common/mjcf.py` | Parse an MJCF file into a skeleton, hinges, couplings, and visual/collision meshes. |
| `_common/rigid.py` | Backend-independent kinematics and mesh posing. |
| `_base.py` | The shared `RigidBodyModel` contract and backend state. |
| `hands/_base.py`, `humanoids/_base.py` | Family constructors, pose parameters and presets. |
| `hands/<name>/_model.py`, `humanoids/<name>/_model.py` | One model: its asset name and common-joint map. |
| `<family>/<name>/numpy.py`, `torch.py`, `jax.py` | Bind the model to an array runtime. |

The loader follows MuJoCo's reading of the file. Skeleton joints are the MJCF
bodies, in body order, and the root body sits at the origin as under a free
joint. Visual meshes are mesh geoms with `contype` and `conaffinity` set to
zero. A hinge named as `joint1` of an `<equality><joint>` follows
`joint1 = polycoef(joint2)`; every other hinge is a pose coordinate, in `qpos`
order. Every hinge angle is therefore one polynomial of one pose coordinate,
so coupled and independent hinges share a single kinematic path.

`ArrayRuntime` owns array construction and model-state materialization. Model
math receives an explicit array namespace, while Torch and JAX wrappers provide
their native module and pytree behavior. Renderer-facing `Trimesh` objects are
created only after array-valued link transforms have been evaluated.

## Adding a model

1. Upload `<name>/assets.zip` and `<name>/LICENSE.md` to the Hugging Face repo.
   The archive holds `<name>.xml` for a humanoid, or `left.xml` and `right.xml`
   for a hand, plus binary or ASCII STL meshes.
2. Add `robot_models/hands/<name>/` or `robot_models/humanoids/<name>/` with `_model.py`, the three backend modules, and
   `assets/poses.npz` (`t_pose`/`a_pose`, or `{side}_flat`/`{side}_rest`).
3. Register it in `_catalog.py` and add a docs page.

The loader also collects contact-enabled geoms and geoms named in explicit
contact pairs as collision meshes. Curved primitives are tessellated for the
mesh interface; the original MJCF keeps their analytic definitions. Visual and
collision meshes use the same skeleton, state materialization and posing path.
Both families expose `collision_meshes` for local geometry and
`forward_meshes(geometry="collision", **params)` for posed meshes.

Loaded state separates skeleton/hinge data from two equivalent mesh groups,
`visual` and `collision`. Each group packs its vertices and faces once; offset
arrays delimit the individual pieces and attach each to a skeleton joint.
Posing transforms those ranges, joins the arrays once and creates one `Trimesh`
per batch element. The per-hand classes contain names and anatomical mappings;
all hands share loading, coupling evaluation, FK and geometry assembly.

Torch materializes both geometry groups as buffers; JAX registers the same data
as a pytree. Internal Torch buffer keys now use `_assets.visual.*` and
`_assets.collision.*`. Serialized model objects and state dictionaries written
with the earlier flat visual layout need to be regenerated. Public model
constructors and numerical methods retain their signatures.

Canonical hand preparation retains original source files and license notices in
the asset bundle. See [hand sources and validation](hand-sources.md).
