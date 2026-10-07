# Allegro

Allegro is the Wonik Robotics Allegro Hand V3. Its skeleton has 21 joints, one per MJCF body, and `hand_pose` has 16 hinge coordinates.

## Setup

The assets come from MuJoCo Menagerie `wonik_allegro` (BSD-2-Clause). They download on first use from the
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository, with the upstream license and a list of changes. To prefetch them:

```bash
robot-models download allegro
```

A manual `model_path` is a directory with `left.xml`, `right.xml` and their meshes.

## Usage

```python
from robot_models.allegro.numpy import AllegroHand

hand = AllegroHand(side="right")
params = hand.get_rest_pose(hands="rest")
```

## Notes

The hand has four fingers and no coupled joints. The common wrist joint is the palm origin at the finger bases; the upstream model has no wrist body.

## API

::: robot_models.allegro.numpy.AllegroHand
