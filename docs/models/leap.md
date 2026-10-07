# LEAP Hand

LEAP Hand is LEAP Hand v1. Its skeleton has 17 joints, one per MJCF body, and `hand_pose` has 16 hinge coordinates.

## Setup

The assets come from MuJoCo Menagerie `leap_hand` (MIT). They download on first use from the
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository, with the upstream license and a list of changes. To prefetch them:

```bash
robot-models download leap
```

A manual `model_path` is a directory with `left.xml`, `right.xml` and their meshes.

## Usage

```python
from robot_models.leap.numpy import LeapHand

hand = LeapHand(side="right")
params = hand.get_rest_pose(hands="rest")
```

## Notes

The hand has four fingers and no coupled joints.

## API

::: robot_models.leap.numpy.LeapHand
