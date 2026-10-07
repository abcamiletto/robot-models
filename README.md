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

- BrainCo Revo 2 robotic hand
- LEAP Hand v1 robotic hand
- Unitree G1 humanoid robot

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
