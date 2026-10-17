# Shadow

Shadow is the Shadow Dexterous Hand E3M5, from the forearm. Its skeleton has 25 joints, one per MJCF body, and `hand_pose` has 24 hinge coordinates.

## Setup

The assets come from MuJoCo Menagerie `shadow_hand` (Apache-2.0). They download on first use from the
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository, with the upstream license and a list of changes. To prefetch them:

```bash
robot-models download shadow
```

A manual `model_path` is a directory with `left.xml`, `right.xml` and their meshes.

## Usage

```python
from robot_models.shadow.numpy import ShadowHand

hand = ShadowHand(side="right")
params = hand.get_rest_pose(hands="rest")
```

## Notes

Every hinge is independent: the J1/J2 tendon coupling of the real hand is actuation, not kinematics.

## API

::: robot_models.shadow.numpy.ShadowHand
