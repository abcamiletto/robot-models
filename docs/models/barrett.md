# Barrett BH280

The default uses Robotnik's BSD-3-Clause ROS description. Its declared URDF mimic relationships are preserved. It has 4 independent coordinates, 8 hinges and 110 skeleton frames (including fixed tactile frames).
The single supplied assembly uses the API side tag `right`; no mirrored assembly
is synthesized. Coordinate counts describe kinematics, rather than motor counts.

```python
from robot_models.hands.barrett.numpy import BarrettHand

hand = BarrettHand()
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
robot-models download barrett
```

::: robot_models.hands.barrett.numpy.BarrettHand
