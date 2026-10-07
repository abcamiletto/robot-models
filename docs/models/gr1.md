# GR1

GR1 is a rigid articulated Fourier GR1T2 humanoid model with STL link meshes
attached to a 33-joint skeleton. It has 32 hinge degrees of freedom: 6 per
leg, 3 in the waist, 7 per arm, and 3 in the head. Dexterous hands are not
included.

## Setup

GR1 downloads automatically on first use from the
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository. The assets are converted to MJCF from the official
Fourier [`Wiki-GRx-Models`](https://github.com/FFTAI/Wiki-GRx-Models) GR1T2
URDF, which is licensed under GPL-3.0. The license and the list of changes are
included with the hosted assets. To prefetch the assets:

```bash
robot-models download gr1
```

When passed manually, `model_path` should contain `gr1.xml` and
`meshes/*.STL`.

## Notes

GR1 does not define `skin_weights`. Use `forward_links()` for link transforms and
`forward_meshes()` for renderable meshes. `get_tpose()` and `get_apose()` return
arm presets for the T-pose and the A-pose.

## API

::: robot_models.gr1.numpy.GR1
