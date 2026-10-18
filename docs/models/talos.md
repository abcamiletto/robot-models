# TALOS

The PAL Robotics TALOS includes two grippers. It has 45 skeleton joints;
`body_pose` has 38 independent hinge coordinates and drives 6 coupled hinges.

Source: MuJoCo Menagerie `pal_talos` (Apache-2.0).
Assets download automatically; see [asset setup](../index.md#assets).

A manual `model_path` is a directory with `talos.xml` and its meshes.

```python
from robot_models.humanoids.talos.numpy import Talos

model = Talos()
params = model.get_tpose()
```

Each gripper has one driven hinge, `gripper_*_inner_double_joint`, that three coupled hinges follow, plus three free fingertip hinges.

## API

::: robot_models.humanoids.talos.numpy.Talos
