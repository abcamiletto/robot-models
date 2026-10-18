# Shadow

The Shadow Dexterous Hand E3M5 starts at the forearm. It has 25 skeleton
joints and 24 independent `hand_pose` coordinates.

Source: MuJoCo Menagerie `shadow_hand` (Apache-2.0).
Assets download automatically; see [asset setup](../index.md#assets).

A manual `model_path` is a directory with `left.xml`, `right.xml` and their meshes.

```python
from robot_models.hands.shadow.numpy import ShadowHand

hand = ShadowHand(side="right")
params = hand.get_rest_pose(preset="rest")
```

Every hinge is independent: the J1/J2 tendon coupling of the real hand is actuation, not kinematics.

## API

::: robot_models.hands.shadow.numpy.ShadowHand
