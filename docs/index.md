# robot-models

`robot-models` provides robot hands and humanoids for NumPy, PyTorch and JAX.

## Install

```bash
uv add robot-models
```

Optional runtime dependencies are available as `robot-models[torch]` and
`robot-models[jax]`. Native MuJoCo attachment uses `robot-models[mujoco]`; see
[adding a hand to a scene](mujoco.md).

## Assets

Assets download automatically on first use when `model_path` is omitted. Use
the CLI to prefetch or configure an explicit location:

```bash
robot-models download g1
robot-models set g1 /path/to/g1
```

## Models

| Model | Description |
| --- | --- |
| [Ability Hand](models/ability.md) | PSYONIC Ability Hand robotic hand |
| [Allegro](models/allegro.md) | Wonik Robotics Allegro robotic hand |
| [Barrett BH280](models/barrett.md) | Barrett BH280 robotic hand |
| [BrainCo](models/brainco.md) | BrainCo Revo 2 robotic hand |
| [EZGripper](models/ezgripper.md) | SAKE EZGripper |
| [G1](models/g1.md) | Unitree G1 humanoid |
| [GR1](models/gr1.md) | Fourier GR1 humanoid |
| [H1](models/h1.md) | Unitree H1 humanoid |
| [Inspire](models/inspire.md) | Inspire RH56 robotic hand |
| [LEAP Hand](models/leap.md) | LEAP Hand v1 robotic hand |
| [Robotiq 3-Finger](models/robotiq.md) | Robotiq 3-Finger Adaptive Gripper |
| [Sharpa Wave](models/sharpa.md) | Sharpa Wave robotic hand |
| [Shadow](models/shadow.md) | Shadow Dexterous Hand E3M5 |
| [T1](models/t1.md) | Booster Robotics T1 humanoid |
| [TALOS](models/talos.md) | PAL Robotics TALOS humanoid |

## Usage

Choose the backend in the import. Hands and humanoids share the geometry API:

```python
from robot_models.humanoids.g1.numpy import G1

model = G1()
params = model.get_rest_pose(batch_dims=(2,))
links = model.forward_links(**params)
collision = model.forward_meshes(geometry="collision", **params)
```

Use `.torch` or `.jax` for other backends. Arrays support any leading batch dimensions.
See the [API reference](api.md) for poses, transforms and joint metadata.
