# SAKE EZGripper

Source: Vikash Kumar's MuJoCo model, maintained by SAKE Robotics (Apache-2.0).
It has 4 hinge coordinates and 5 skeleton frames; the side tag is `right`.
Native tendons, contacts and the actuator are retained. FK does not solve tendon dynamics.

```python
from robot_models.hands.ezgripper.numpy import EZGripper

hand = EZGripper()
params = hand.get_rest_pose(preset="rest")
meshes = hand.visual.pose(hand.forward_skeleton(**params))
```

Assets download automatically. See [sources and licenses](../hand-sources.md).

`flat` and `rest` use zero angles clipped to source limits.

## API

::: robot_models.hands.ezgripper.numpy.EZGripper
