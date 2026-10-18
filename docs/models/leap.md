# LEAP Hand

LEAP Hand v1 has 17 skeleton joints and 16 independent `hand_pose` coordinates.

Source: MuJoCo Menagerie `leap_hand` (MIT).
Assets download automatically; see [asset setup](../index.md#assets).

A manual `model_path` is a directory with `left.xml`, `right.xml` and their meshes.

```python
from robot_models.hands.leap.numpy import LeapHand

hand = LeapHand(side="right")
params = hand.get_rest_pose(hands="rest")
```

The hand has four fingers and no coupled joints.

## API

::: robot_models.hands.leap.numpy.LeapHand
