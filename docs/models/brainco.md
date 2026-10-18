# BrainCo

The BrainCo Revo 2 has 17 skeleton joints. `hand_pose` has 6 independent hinge
coordinates and drives 5 coupled hinges.

Source: the official BrainCo Revo 2 MuJoCo package (see the hosted `LICENSE.md`).
Assets download automatically; see [asset setup](../index.md#assets).

A manual `model_path` is a directory with `left.xml`, `right.xml` and their meshes.

```python
from robot_models.hands.brainco.numpy import BrainCoHand

hand = BrainCoHand(side="right")
params = hand.get_rest_pose(preset="rest")
```

Each distal hinge follows its proximal hinge, as the BrainCo actuators drive the proximal joints. The hosted XML states these equalities in MuJoCo's `joint1 = poly(joint2)` order and keeps the fingertips rigid.

## API

::: robot_models.hands.brainco.numpy.BrainCoHand
