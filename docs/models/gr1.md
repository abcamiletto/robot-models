# GR1

The Fourier GR1T2 has 33 skeleton joints and 32 `body_pose` coordinates,
without dexterous hands.

Source: Fourier `Wiki-GRx-Models`, converted from URDF to MJCF (GPL-3.0).
Assets download automatically; see [asset setup](../index.md#assets).

A manual `model_path` is a directory with `gr1.xml` and its meshes.

```python
from robot_models.humanoids.gr1.numpy import GR1

model = GR1()
params = model.get_rest_pose(preset="t_pose")
```

Data recorded with the robosuite GR1 variant (RoboCasa, DexMimicGen, GR00T) uses another joint order and slightly different offsets.

## API

::: robot_models.humanoids.gr1.numpy.GR1
