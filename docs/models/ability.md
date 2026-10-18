# Ability Hand

The PSYONIC Ability Hand, large size, has 11 skeleton joints. `hand_pose` has
6 independent hinge coordinates and drives 4 coupled hinges.

Source: the official PSYONIC MuJoCo model in `psyonicinc/ability-hand-api` (MIT).
Assets download automatically; see [asset setup](../index.md#assets).

A manual `model_path` is a directory with `left.xml`, `right.xml` and their meshes.

```python
from robot_models.hands.ability.numpy import AbilityHand

hand = AbilityHand(side="right")
params = hand.get_rest_pose(preset="rest")
```

Each finger's distal `pip` hinge follows its `mcp` hinge through `pip = 0.72349796 + 1.05851325 * mcp`, the linear four-bar approximation of the official simulator. It differs from the exact linkage by at most 2.4 degrees.

## API

::: robot_models.hands.ability.numpy.AbilityHand
