# Add a hand to MuJoCo

Install `robot-models[mujoco]`. `add_hand` attaches a hand to an uncompiled
`mujoco.MjSpec` and returns `(mount_frame, root_body)`:

```python
import mujoco
from robot_models import add_hand

scene = mujoco.MjSpec()
mount, root = add_hand(scene, "shadow", side="right")
mount.pos = [0, 0, 0.4]
model = scene.compile()
```

All eleven [hands](hand-sources.md) are supported. Barrett, Robotiq and EZGripper use
only the `right` assembly tag; the others provide both sides. Assets download
automatically; MANO's are built locally. Pass `model_path` to use a local bundle.
For MANO, `shape` and `conform_to` build the hand of `ManoRobot(shape=..., conform_to=...)`:
its joints, meshes and link inertias, before attachment.

## Mount on a robot

Pass a frame or its name. Names receive the prefix `{name}_{side}_` by default;
use a unique `prefix` for repeated copies:

```python
scene = mujoco.MjSpec.from_file("robot.xml")
wrist = scene.body("wrist").add_frame(name="hand_mount")
add_hand(scene, "allegro", side="left", frame=wrist, prefix="gripper_")
model = scene.compile()
```

The root is fixed by default. Set the returned frame's `pos` and `quat` to place
it; use `root` to access the attached body without constructing a prefixed name.
For a floating hand, use `mount, root = add_hand(scene, "shadow", free=True)` at
the world level. Its joint block starts with seven root coordinates (xyz, wxyz),
followed by hinges. A free joint cannot be mounted under another moving body.

MuJoCo uses **Z-up and wxyz quaternions**; model FK uses Y-up. With
`M = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]`, convert a model-frame translation `t`
and rotation `R` to MuJoCo with `M.T @ t` and `M.T @ R @ M`.

`to_qpos` returns a free hand's qpos block: its root, then every hinge. Its root is in
MuJoCo's world turned from the model frame by default; pass `scene_transform`, the
4×4 rigid transform `(A, b)` from the frame of `global_rotation` and
`global_translation` to the scene's world, to place it anywhere: the root goes to
`A @ t + b`, oriented `A @ R @ M`. `qpos_joint_names` names the hinges in order,
coupled ones included, so that each one's address is found by name:

```python
mount, root = add_hand(scene, "mano", prefix="h1-", free=True)
model = scene.compile()
qpos = hand.to_qpos(pose, global_rotation=r, global_translation=t, scene_transform=object_to_world)
start = model.jnt_qposadr[model.body_jntadr[model.body(root.name).id]]
data.qpos[start : start + 7] = qpos[:7]
for name, angle in zip(hand.qpos_joint_names, qpos[7:]):
    data.qpos[model.joint("h1-" + name).qposadr[0]] = angle
```

## Physics

Attachment retains native meshes, collision shapes, joints, actuators, equalities
and tendons. The parent scene controls global solver options and can override
mass scaling; child `settotalmass` is resolved into body inertias first.
Child options and legacy allocation limits are matched to the scene before
attachment, so mounting emits no option-conflict warnings or log files.

Barrett and Robotiq have placeholder inertias inferred from visual mesh volume
at 1000 kg/m³. Their dynamics, actuation and contact settings need to be configured
for the application. See [collision behavior](collisions.md).

Finite [joint torque limits](hand-control.md#torque-sources) are applied to existing joint actuators.
