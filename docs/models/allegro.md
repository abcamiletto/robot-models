# Allegro

Allegro is a rigid articulated model of the Wonik Robotics Allegro Hand V3
using the MuJoCo Menagerie `wonik_allegro` XML and STL assets.

## Setup

Allegro downloads from the public
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository on first use. To prefetch the assets:

```bash
robot-models download allegro
```

When passed manually, `model_path` should contain `left.xml`, `right.xml`, and
`meshes/{left,right}/*.STL`.

The original BSD-2-Clause license is included with the hosted assets.

## Usage

```python
from robot_models.allegro.numpy import AllegroHand

hand = AllegroHand(side="right")
```

## Notes

The model exposes 16 hinge joints for each hand: four joints for each of the
index, middle, and ring fingers and four for the thumb. Allegro has no pinky.
`Joint.*_WRIST` maps to the palm frame, which is at the base of the fingers.

## API

::: robot_models.allegro.numpy.AllegroHand
