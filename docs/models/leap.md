# LEAP Hand

LEAP Hand is a rigid articulated model of the CMU LEAP Hand v1 robotic hand
using the MuJoCo Menagerie `leap_hand` XML and meshes.

## Setup

LEAP Hand downloads from the public
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository on first use. To prefetch the assets:

```bash
robot-models download leap
```

When passed manually, `model_path` should contain `left.xml`, `right.xml`, and
`meshes/{left,right}/*.STL`.

The original MuJoCo Menagerie MIT license is included with the hosted assets.

## Usage

```python
from robot_models.leap.numpy import LeapHand

hand = LeapHand(side="right")
```

## Notes

The model exposes the 16 actuated LEAP joints for each hand: MCP flexion, MCP
rotation, PIP, and DIP for the index, middle, and ring fingers, and CMC, axial,
MCP, and IP for the thumb. LEAP Hand has no pinky and no coupled joints.

## API

::: robot_models.leap.numpy.LeapHand
