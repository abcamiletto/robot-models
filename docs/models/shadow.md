# Shadow

Shadow is a rigid articulated model of the Shadow Dexterous Hand E3M5 using the
MuJoCo Menagerie `shadow_hand` assets.

## Setup

Shadow downloads from the public
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository on first use. To prefetch the assets:

```bash
robot-models download shadow
```

When passed manually, `model_path` should contain `left.xml`, `right.xml`, and
`meshes/{left,right}/*.STL`.

The assets come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand)
under the Apache 2.0 license, copyright Shadow Robot Company. The hosted
license notes the changes, mainly the OBJ to STL mesh conversion.

## Usage

```python
from robot_models.shadow.numpy import ShadowHand

hand = ShadowHand(side="right")
```

## Notes

The model exposes all 24 hinge joints of each hand as `hand_pose`: two wrist
joints, five thumb joints, four joints for each of the index, middle, and ring
fingers, and five little-finger joints including the metacarpal. The real hand
couples J1 and J2 of each finger through one tendon. This coupling is an
actuation detail, so both joints stay independent in the model.

## API

::: robot_models.shadow.numpy.ShadowHand
