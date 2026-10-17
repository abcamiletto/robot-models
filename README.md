# robot-models

`robot-models` provides rigid articulated robot and musculoskeletal models for
NumPy, PyTorch, and JAX. It is independent of `body-models`.

Documentation: https://abcamiletto.github.io/robot-models/

## Install

```bash
uv add robot-models
uv add "robot-models[torch]"
uv add "robot-models[jax]"
```

Public assets download automatically on first use and can be prefetched with
`robot-models download MODEL`.

## Quick start

```python
from robot_models.g1.torch import G1

model = G1()
params = model.get_rest_pose(batch_dims=(1,))

links = model.forward_links(**params)
skeleton = model.forward_skeleton(**params)
meshes = model.forward_meshes(**params)
```

The equivalent NumPy and JAX classes live in `robot_models.g1.numpy` and
`robot_models.g1.jax`. Torch models are `torch.nn.Module` instances.

## Models

- Booster Robotics T1 humanoid robot
- BrainCo Revo 2 robotic hand
- Fourier GR1 humanoid robot
- Inspire RH56 robotic hand
- LEAP Hand v1 robotic hand
- PAL Robotics TALOS humanoid robot
- PSYONIC Ability Hand robotic hand
- Sharpa Wave robotic hand
- Shadow Dexterous Hand E3M5
- Unitree G1 humanoid robot
- Unitree H1 humanoid robot
- Wonik Robotics Allegro robotic hand
- Barrett BH280, Robotiq 3-Finger and SAKE EZGripper

Each hand has one canonical model from its manufacturer, MuJoCo Menagerie or
an established ROS description. See the [hand source and validation record](docs/hand-sources.md).

## Teasers

Render separate humanoid and hand lineups, in the style of Body Models:

![Humanoid lineup: Unitree G1, Fourier GR1, Unitree H1, Booster T1 and PAL TALOS.](docs/assets/teasers/humanoids.png)

![Hand lineup: Ability, Allegro, Barrett, BrainCo, EZGripper, Inspire, LEAP, Robotiq 3F, Sharpa and Shadow.](docs/assets/teasers/hands.png)

Both images have transparent backgrounds. Models are scaled to equal heights for
display; the lineups do not compare physical size. Regenerate the README images:

```bash
uv run --script scripts/render_teaser.py --output-dir docs/assets/teasers
```

This provisions an isolated Python 3.13/Blender environment. See
[rendering options and model selection](docs/teasers.md) for GPU rendering,
resolution and editable Blender scenes.

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest -m fast
```

## License

The code is licensed under the Apache License 2.0 (see `LICENSE`). Model assets
are licensed separately by their upstream projects — see the documentation and
upstream model pages for model-specific terms.
