# Sharpa Wave

Sharpa Wave has 23 skeleton joints and 22 independent `hand_pose` coordinates:
five thumb hinges, four per finger and one extra pinky CMC hinge.

Source: official Sharpa MJCF (Apache-2.0), using the standard palm-rooted hands.
Assets download automatically with licenses and notices; see
[pinned source and terms](../hand-sources.md).
A local `model_path` must contain `left.xml`, `right.xml` and their mesh folders.

```python
from robot_models.hands.sharpa.numpy import SharpaHand

hand = SharpaHand(side="right")  # also supports side="left"
params = hand.get_rest_pose(hands="rest")
skeleton = hand.forward_skeleton(**params)
meshes = hand.forward_meshes(**params)
```

The `flat` preset is the zero joint pose. The `rest` preset uses 0.2 radians
for flexion hinges (CMC FE, MCP FE, IP, PIP, and DIP), 0.05 radians for the
pinky CMC, and zero for abduction/adduction. These are package reference poses,
not manufacturer-calibrated resting poses.

Common joints map to thumb MC/PP/DP and finger PP/MP/DP segments. The full
skeleton also includes the virtual links of the two-axis CMC and MCP joints.

## API

::: robot_models.hands.sharpa.numpy.SharpaHand
