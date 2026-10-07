# Inspire

Inspire is the Inspire RH56 hand as mounted on the Unitree G1. Its skeleton has 13 joints, one per MJCF body, and `hand_pose` has 6 hinge coordinates; 6 more hinges are coupled to them.

## Setup

The assets come from Unitree's `unitree_ros` description, converted from URDF to MJCF (BSD-3-Clause). They download on first use from the
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository, with the upstream license and a list of changes. To prefetch them:

```bash
robot-models download inspire
```

A manual `model_path` is a directory with `left.xml`, `right.xml` and their meshes.

## Usage

```python
from robot_models.inspire.numpy import InspireHand

hand = InspireHand(side="right")
params = hand.get_rest_pose(hands="rest")
```

## Notes

Intermediate and distal hinges follow their drivers through the URDF mimic ratios. The root is the hand base; on a G1 it sits 0.0415 m along the `*_wrist_yaw_link` x axis.

## API

::: robot_models.inspire.numpy.InspireHand
