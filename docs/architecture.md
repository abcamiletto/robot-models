# Architecture

Every model is a MuJoCo MJCF file hosted on Hugging Face, read by one loader:

| Module | Responsibility |
| --- | --- |
| `_assets.py` | Resolve a model's asset directory: explicit path, config, cache, or download. |
| `_common/mjcf.py` | Parse an MJCF file into a skeleton, hinges, couplings, and visual meshes. |
| `_common/rigid.py` | Backend-independent kinematics and mesh posing. |
| `_base.py` | `RigidBodyModel` and its two families, `Humanoid` and `Hand`. |
| `<name>/_model.py` | One model: its asset name and common-joint map. |
| `<name>/numpy.py`, `torch.py`, `jax.py` | Bind the model to an array runtime. |

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
2. Add `robot_models/<name>/` with `_model.py`, the three backend modules, and
   `assets/poses.npz` (`t_pose`/`a_pose`, or `{side}_flat`/`{side}_rest`).
3. Register it in `_catalog.py` and add a docs page.
