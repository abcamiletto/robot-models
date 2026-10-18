# SMPL

`SmplRobot` is SMPL (Loper et al., SIGGRAPH Asia 2015) as a rigid humanoid:
Body Models' `create_model("smpl", gender=gender)`. It has SMPL's 24 skeleton
joints, in depth-first order, and 69 `body_pose` coordinates: three hinges per
joint, about the body frame's x, y and z axes in that order, without limits.

```python
from robot_models.humanoids.smpl.numpy import SmplRobot

body = SmplRobot(gender="neutral")
params = body.from_body_models(**smpl_params)
skeleton = body.forward_skeleton(**params)
links = body.forward_meshes(**params)  # rigid links
skin = body.forward_vertices(**params)  # SMPL's skin, exactly; faces in body.skin_faces
```

SMPL is built like [MANO](mano.md): locally, from your own files as Body Models
configures them (`body-models set smpl-neutral PATH`, likewise `male` and `female`),
in about two minutes, never downloaded or redistributed. `shape`, the conversion
from and to Body Models parameters (its `pelvis_rotation` folds into
`global_rotation`), the rigid links of the rest skin and their convex pieces
(about 960) follow the same principle. Links have 1000 kg/m³ density; the root has
a free joint; the MJCF has no actuators.

## API

::: robot_models.humanoids.smpl.numpy.SmplRobot
