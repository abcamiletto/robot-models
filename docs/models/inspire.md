# Inspire

Inspire is a rigid articulated model of the Inspire Robots RH56 dexterous hand
(DFQ variant), the hand that Unitree mounts on the G1 and H1 humanoids. The
assets come from the Unitree
[`unitree_ros`](https://github.com/unitreerobotics/unitree_ros) G1 description.

## Setup

Inspire downloads from the public
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository on first use. To prefetch the assets:

```bash
robot-models download inspire
```

When passed manually, `model_path` should contain `left.xml`, `right.xml`, and
`meshes/{left,right}/*.STL`.

## License

The upstream URDF and STL files are licensed under the BSD 3-Clause License
by Unitree Robotics. The hosted assets convert the URDF files to MuJoCo XML and
turn the URDF mimic joints into MuJoCo joint equality constraints. The license
text and a list of changes are included with the hosted assets.

## Usage

```python
from robot_models.inspire.numpy import InspireHand

hand = InspireHand(side="right")
```

## Notes

The model exposes the six actuated RH56 joints for each hand: thumb yaw, thumb
pitch, and the proximal joints for index, middle, ring, and pinky. The thumb
intermediate and distal joints follow thumb pitch with ratios 1.6 and 2.4. Each
finger intermediate joint follows its proximal joint with ratio 1.

The root frame is the hand base, rotated to align with the G1
`*_wrist_yaw_link` frame. On the G1, the hand base is 0.0415 m along the wrist
x axis.

## API

::: robot_models.inspire.numpy.InspireHand
