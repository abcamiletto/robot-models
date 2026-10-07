# Ability Hand

Ability Hand is the PSYONIC Ability Hand (large size). Its skeleton has 11 joints, one per MJCF body, and `hand_pose` has 6 hinge coordinates; 4 more hinges are coupled to them.

## Setup

The assets come from the official PSYONIC MuJoCo model in `psyonicinc/ability-hand-api` (MIT). They download on first use from the
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository, with the upstream license and a list of changes. To prefetch them:

```bash
robot-models download ability
```

A manual `model_path` is a directory with `left.xml`, `right.xml` and their meshes.

## Usage

```python
from robot_models.ability.numpy import AbilityHand

hand = AbilityHand(side="right")
params = hand.get_rest_pose(hands="rest")
```

## Notes

Each finger's distal `pip` hinge follows its `mcp` hinge through `pip = 0.72349796 + 1.05851325 * mcp`, the linear four-bar approximation of the official simulator. It differs from the exact linkage by at most 2.4 degrees.

## API

::: robot_models.ability.numpy.AbilityHand
