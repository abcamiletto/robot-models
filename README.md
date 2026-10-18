# robot-models

Robot hands and humanoids for NumPy, PyTorch, and JAX.
[Documentation](https://abcamiletto.github.io/robot-models/).

## Install

```bash
uv add robot-models              # NumPy
uv add "robot-models[torch]"     # PyTorch
uv add "robot-models[jax]"       # JAX
```

Assets download from [Hugging Face](https://huggingface.co/abcamiletto/robot-models)
on first use. To prefetch a model, run `robot-models download MODEL`.

## Usage

```python
from robot_models.hands.shadow.torch import ShadowHand
from robot_models.humanoids.g1.torch import G1

hand = ShadowHand(side="right")
humanoid = G1()
params = hand.get_rest_pose(batch_dims=(2,))

skeleton = hand.forward_skeleton(**params)
visual = hand.forward_meshes(**params)
collision = hand.forward_meshes(geometry="collision", **params)
```

Choose the backend with `.numpy`, `.torch` or `.jax` in the import.
Torch models are `torch.nn.Module` instances. Both families provide
[visual and collision meshes](docs/collisions.md).

## Models

| Family | Model identifiers |
| --- | --- |
| `robot_models.humanoids` | `g1`, `gr1`, `h1`, `t1`, `talos` |
| `robot_models.hands` | `ability`, `allegro`, `barrett`, `brainco`, `ezgripper`, `inspire`, `leap`, `robotiq-3finger`, `sharpa`, `shadow` |

Use `list_models(family="hands")` or `list_models(family="humanoids")` to list models.
Each hand uses one manufacturer, Menagerie or established ROS description.
See [sources and licenses](docs/hand-sources.md).
Use [joint semantics, closing presets and torque limits](docs/hand-control.md) for hand-independent control.

## MuJoCo

```bash
uv add "robot-models[mujoco]"
```

```python
import mujoco
from robot_models import add_hand

scene = mujoco.MjSpec()
mount, root = add_hand(scene, "shadow", side="right")
mount.pos = [0, 0, 0.4]
model = scene.compile()
```

`add_hand` supports all ten hands and retains their native collision shapes.
Barrett and Robotiq need application-specific dynamics and actuation.
See [mounting and coordinate frames](docs/mujoco.md).

## Teasers

![Unitree G1, Fourier GR1, Unitree H1, Booster T1 and PAL TALOS.](docs/assets/teasers/humanoids.png)

![Ability, Allegro, Barrett, BrainCo, EZGripper, Inspire, LEAP, Robotiq 3F, Sharpa and Shadow.](docs/assets/teasers/hands.png)

The backgrounds are transparent. Models are scaled to equal heights for display.
Regenerate both images:

```bash
uv run --script scripts/render_teaser.py --output-dir docs/assets/teasers
```

See [rendering options](docs/teasers.md).

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest -m fast
```

## License

Code: [Apache-2.0](LICENSE). Assets use their upstream licenses; see each model's documentation.
