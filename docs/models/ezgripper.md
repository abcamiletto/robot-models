# SAKE EZGripper

The default uses the Apache-2.0 MuJoCo model by Vikash Kumar, linked on SAKE Robotics' support page. Native tendons, contacts and the actuator remain in the source/runtime XML; FK does not solve tendon dynamics. It has 4 independent hinge pose coordinates and 5 skeleton frames.
The single supplied assembly uses the API side tag `right`; no mirrored assembly
is synthesized. Coordinate counts describe kinematics, rather than motor counts.

```python
from robot_models.ezgripper.numpy import EZGripper

hand = EZGripper()
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
robot-models download ezgripper
```

::: robot_models.ezgripper.numpy.EZGripper
