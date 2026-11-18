# Anatomical SOMA

NVIDIA's Anatomical SOMA is a simulated human built from SOMA23, with the same
proportions and masses (1.74 m, 42.5 kg). Each of its 15 ball joints is replaced by a
serial chain of three hinges, each with its own limits, and the knees, elbows, toes and
upper neck have one hinge each. It has 53 skeleton joints, including the hinge carrier
bodies, and 52 `body_pose` coordinates.

Source: NVlabs ProtoMotions `anatomic_soma.xml` (Apache-2.0), with joint ranges
converted to radians.
Assets download automatically; see [asset setup](../index.md#assets).

A manual `model_path` is a directory with `anatomic-soma.xml`.

```python
from robot_models.humanoids.anatomic_soma.numpy import AnatomicSoma

model = AnatomicSoma()
params = model.get_rest_pose(preset="a_pose")
frames = model.forward_skeleton(**params)
mesh = model.collision.pose(frames)[0]
```

The geometry is 39 collision primitives: capsules, a box head and rails along the
feet. The model has no visual meshes, so use `collision`. Zero pose is the asset's
T-pose.

Hinge suffixes keep SOMA's axis letters: `_x` is pitch (flexion), `_y` roll
(abduction) and `_z` twist. `body_pose` follows the upstream asset's MuJoCo `qpos`
order, and a body's hinges rotate in that order. Carrier bodies are named by their position in a chain, not by the hinge
they hold. `Spine1_x_link`, for example, carries `Spine1_z`; see the upstream
[robot guide](https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/robots/anatomic_soma.rst).

## API

::: robot_models.humanoids.anatomic_soma.numpy.AnatomicSoma
