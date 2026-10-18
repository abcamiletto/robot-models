# Robotiq 3-Finger

The Robotiq-authored ROS description (BSD-2-Clause) has 14 skeleton frames and
11 independent hinge coordinates. Original xacro and DAE visuals are retained.
The supplied assembly uses the side tag `right`.

```python
from robot_models.hands.robotiq_3finger.numpy import Robotiq3Finger

hand = Robotiq3Finger()
params = hand.get_rest_pose(hands="rest")
meshes = hand.forward_meshes(**params)
```

Assets download automatically. See [sources and licenses](../hand-sources.md).

`flat` and `rest` use zero angles clipped to source limits.

## API

::: robot_models.hands.robotiq_3finger.numpy.Robotiq3Finger
