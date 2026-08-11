# Architecture

Each model family is self-contained in `robot_models/<name>/`:

| File | Responsibility |
| --- | --- |
| `_io.py` | Resolve assets and load immutable NumPy model data. |
| `_core.py` | Backend-independent kinematics. |
| `_model.py` | Public model contract and forward orchestration. |
| `numpy.py`, `torch.py`, `jax.py` | Bind the model to an array runtime. |

`RigidBodyModel` owns metadata, pose packing, MuJoCo `qpos` conversion, link
attachment, and mesh projection. Model-local cores retain their distinct
kinematics: BrainCo coupled joints, G1 hinge axes, SmplHumanoid Euler controls,
and MyoFullBody mixed hinge/slide joints.

`ArrayRuntime` owns array construction and model-state materialization. Model
math receives an explicit array namespace, while Torch and JAX wrappers provide
their native module and pytree behavior. Renderer-facing `Trimesh` objects are
created only after array-valued link transforms have been evaluated.
