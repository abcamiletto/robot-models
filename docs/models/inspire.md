# Inspire

The Inspire RH56, as mounted on the Unitree G1, has 13 skeleton joints.
`hand_pose` has 6 independent hinge coordinates and drives 6 coupled hinges.

Source: Unitree's `unitree_ros` description, converted from URDF to MJCF (BSD-3-Clause).
Assets download automatically; see [asset setup](../index.md#assets).

A manual `model_path` is a directory with `left.xml`, `right.xml` and their meshes.

```python
from robot_models.hands.inspire.numpy import InspireHand

hand = InspireHand(side="right")
params = hand.get_rest_pose(preset="rest")
```

Intermediate and distal hinges follow their drivers through the URDF mimic ratios. The root is the hand base; on a G1 it sits 0.0415 m along the `*_wrist_yaw_link` x axis.

## API

::: robot_models.hands.inspire.numpy.InspireHand
