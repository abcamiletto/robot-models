# Robotiq 3-Finger

The default uses the historical BSD-2-Clause ROS description authored by Robotiq. The original xacro and DAE visuals are retained. It has 11 independent hinge coordinates and 14 skeleton frames.
The single supplied assembly uses the API side tag `right`; no mirrored assembly
is synthesized. Coordinate counts describe kinematics, rather than motor counts.

```python
from robot_models.hands.robotiq_3finger.numpy import Robotiq3Finger

hand = Robotiq3Finger()
params = hand.get_rest_pose(hands="rest")
meshes = hand.forward_meshes(**params)
```

Assets download from our Hugging Face repository, retaining original source files,
full upstream license text, provenance and hashes. See the
[source and validation record](../hand-sources.md) for pinned revisions,
reproducible preparation and independent MuJoCo comparisons.

`flat` and `rest` return zero clipped to source limits, rather than a motor
equilibrium or optimizer initialization.

```bash
robot-models download robotiq-3finger
```

## API

::: robot_models.hands.robotiq_3finger.numpy.Robotiq3Finger
