# GR1

GR1 is the Fourier GR1T2 humanoid without dexterous hands. Its skeleton has 33 joints, one per MJCF body, and `body_pose` has 32 hinge coordinates.

## Setup

The assets come from Fourier `Wiki-GRx-Models`, converted from URDF to MJCF (GPL-3.0). They download on first use from the
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository, with the upstream license and a list of changes. To prefetch them:

```bash
robot-models download gr1
```

A manual `model_path` is a directory with `gr1.xml` and its meshes.

## Usage

```python
from robot_models.gr1.numpy import GR1

model = GR1()
params = model.get_tpose()
```

## Notes

Data recorded with the robosuite GR1 variant (RoboCasa, DexMimicGen, GR00T) uses another joint order and slightly different offsets.

## API

::: robot_models.gr1.numpy.GR1
