# robot-models

`robot-models` provides rigid articulated models with NumPy, PyTorch, and JAX
runtimes. It is a standalone package and does not depend on `body-models`.

## Install

```bash
uv add robot-models
```

Optional runtime dependencies are available as `robot-models[torch]` and
`robot-models[jax]`.

## Assets

Assets download automatically on first use when `model_path` is omitted. Use
the CLI to prefetch or configure an explicit location:

```bash
robot-models download g1
robot-models set g1 /path/to/g1
```

## Models

| Model | Scope | Setup |
| --- | --- | --- |
| [BrainCo](models/brainco.md) | BrainCo Revo 2 robotic hand | auto-download |
| [G1](models/g1.md) | Unitree G1 humanoid | auto-download |
| [MyoFullBody](models/myofullbody.md) | MuJoCo musculoskeletal model | auto-download |
| [SmplHumanoid](models/smpl-humanoid.md) | SMPL-compatible MJCF humanoids | auto-download |

## Usage

Each model has `numpy`, `torch`, and `jax` modules. All models derive from
`RigidBodyModel` and expose skeleton transforms, link transforms, link-local
meshes, posed meshes, joint metadata, and MuJoCo `qpos` conversion.

```python
from robot_models.g1.numpy import G1

model = G1()
params = model.get_rest_pose(batch_dims=(2,))
links = model.forward_links(**params)
```

Array shapes support arbitrary leading batch dimensions. Underscore-prefixed
modules are private implementation details.
