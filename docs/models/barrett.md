# Barrett BH280

Robotnik's BH280 ROS description (BSD-3-Clause) has 110 skeleton frames, including
fixed tactile frames. Eight hinges use 4 independent coordinates and the source
URDF mimic rules. The supplied assembly uses the side tag `right`.

```python
from robot_models.hands.barrett.numpy import BarrettHand

hand = BarrettHand()
params = hand.get_rest_pose(preset="rest")
meshes = hand.visual.pose(hand.forward_skeleton(**params))
```

Assets download automatically. See [sources and licenses](../hand-sources.md).

`flat` and `rest` use zero angles clipped to source limits.

## API

::: robot_models.hands.barrett.numpy.BarrettHand
