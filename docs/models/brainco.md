# BrainCo

BrainCo is the BrainCo Revo 2 hand. Its skeleton has 17 joints, one per MJCF body, and `hand_pose` has 6 hinge coordinates; 5 more hinges are coupled to them.

## Setup

The assets come from the official BrainCo Revo 2 MuJoCo package (see the hosted `LICENSE.md`). They download on first use from the
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository, with the upstream license and a list of changes. To prefetch them:

```bash
robot-models download brainco
```

A manual `model_path` is a directory with `left.xml`, `right.xml` and their meshes.

## Usage

```python
from robot_models.brainco.numpy import BrainCoHand

hand = BrainCoHand(side="right")
params = hand.get_rest_pose(hands="rest")
```

## Notes

Each distal hinge follows its proximal hinge, as the BrainCo actuators drive the proximal joints. The hosted XML states these equalities in MuJoCo's `joint1 = poly(joint2)` order and keeps the fingertips rigid.

## API

::: robot_models.brainco.numpy.BrainCoHand
