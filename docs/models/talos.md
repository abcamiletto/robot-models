# TALOS

TALOS is the PAL Robotics TALOS humanoid with its two grippers. Its skeleton has 45 joints, one per MJCF body, and `body_pose` has 38 hinge coordinates; 6 more hinges are coupled to them.

## Setup

The assets come from MuJoCo Menagerie `pal_talos` (Apache-2.0). They download on first use from the
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository, with the upstream license and a list of changes. To prefetch them:

```bash
robot-models download talos
```

A manual `model_path` is a directory with `talos.xml` and its meshes.

## Usage

```python
from robot_models.talos.numpy import Talos

model = Talos()
params = model.get_tpose()
```

## Notes

Each gripper has one driven hinge, `gripper_*_inner_double_joint`, that three coupled hinges follow, plus three free fingertip hinges.

## API

::: robot_models.talos.numpy.Talos
