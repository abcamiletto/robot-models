# Add a hand to MuJoCo

Install `robot-models[mujoco]`. `add_hand` attaches a hand to an uncompiled
`mujoco.MjSpec` and returns its mount frame:

```python
import mujoco
from robot_models import add_hand

scene = mujoco.MjSpec()
mount = add_hand(scene, "shadow", side="right")
mount.pos = [0, 0, 0.4]
model = scene.compile()
```

All ten [hands](hand-sources.md) are supported. Barrett, Robotiq and EZGripper use
only the `right` assembly tag; the others provide both sides. Assets download
automatically. Pass `model_path` to use a local bundle.

## Mount on a robot

Pass a frame or its name. Names receive the prefix `{name}_{side}_` by default;
use a unique `prefix` for repeated copies:

```python
scene = mujoco.MjSpec.from_file("robot.xml")
wrist = scene.body("wrist").add_frame(name="hand_mount")
add_hand(scene, "allegro", side="left", frame=wrist, prefix="gripper_")
model = scene.compile()
```

The root is fixed to the mount with identity local rotation. Set the returned
frame's `pos` and `quat` to place it. The helper adds no floating joint.

MuJoCo uses **Z-up and wxyz quaternions**; model FK uses Y-up. With
`M = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]`, convert a model-frame translation `t`
and rotation `R` to MuJoCo with `M.T @ t` and `M.T @ R @ M`.

## Physics

Attachment retains native meshes, collision shapes, joints, actuators, equalities
and tendons. The parent scene controls global solver options and can override
mass scaling; child `settotalmass` is resolved into body inertias first.
MuJoCo can warn about different child options.

Barrett and Robotiq have placeholder inertias inferred from visual mesh volume
at 1000 kg/m³. Their dynamics, actuation and contact settings need to be configured
for the application. See [collision behavior](collisions.md).
