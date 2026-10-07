# Ability Hand

Ability Hand is a rigid articulated model of the PSYONIC Ability Hand (large
size) using the official MuJoCo XML and STL assets from
[`psyonicinc/ability-hand-api`](https://github.com/psyonicinc/ability-hand-api).

## Setup

Ability Hand downloads from the public
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository on first use. To prefetch the assets:

```bash
robot-models download ability
```

When passed manually, `model_path` should contain `left.xml`, `right.xml`, and
`meshes/{left,right}/*.STL`.

The upstream assets are MIT licensed. The license and a list of changes are
included with the hosted assets.

## Usage

```python
from robot_models.ability.numpy import AbilityHand

hand = AbilityHand(side="right")
```

## Notes

The model exposes the six actuated joints for each hand: thumb rotator
(`thumb_cmc`), thumb flexor (`thumb_mcp`), and the proximal `mcp` joints for
index, middle, ring, and pinky. The thumb has no coupled joint.

Each finger has a coupled distal `pip` joint. The real hand uses a nonlinear
four-bar linkage. This model uses the linear approximation from the official
PSYONIC MuJoCo simulator, `pip = 0.72349796 + 1.05851325 * mcp`. It is stored
as a MuJoCo joint equality. Over the `mcp` range, the approximation differs
from the exact four-bar angle by at most 2.4 degrees.

## API

::: robot_models.ability.numpy.AbilityHand
