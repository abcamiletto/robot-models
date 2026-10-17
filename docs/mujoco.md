# Add a hand to MuJoCo

Install the optional native MuJoCo dependency:

```bash
uv add "robot-models[mujoco]"
```

`add_hand` attaches any canonical hand to an uncompiled `mujoco.MjSpec`, using
its usual asset download/configuration. It returns the attachment frame:

```python
import mujoco
from robot_models import add_hand

scene = mujoco.MjSpec()
mount = add_hand(scene, "shadow", side="right")
mount.pos = [0, 0, 0.4]
model = scene.compile()
data = mujoco.MjData(model)
```

Supported identifiers are `ability`, `allegro`, `barrett`, `brainco`, `ezgripper`,
`inspire`, `leap`, `robotiq-3finger`, `sharpa` and `shadow`. Barrett, Robotiq and
EZGripper supply only the `right` assembly tag; the others supply both sides.

## Mounting on an existing robot

Pass an existing MuJoCo frame, or its name. Names are prefixed automatically with
`{name}_{side}_`; use a distinct `prefix` when adding the same hand and side twice.

```python
scene = mujoco.MjSpec.from_file("robot.xml")
wrist = scene.body("wrist").add_frame(name="hand_mount")
add_hand(scene, "allegro", side="left", frame=wrist, prefix="gripper_")
model = scene.compile()
```

The hand's root is fixed to the mount with identity local rotation, matching the
root convention of `create_model`. The returned frame's `pos` and `quat` set its
local placement. MuJoCo uses **Z-up and wxyz quaternions**; robot-models' FK uses
Y-up. For a MuJoCo vector ``v``, the model-frame vector is ``M @ v``, where
``M = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]``. A model-frame translation ``t`` and
rotation matrix ``R`` become mount position ``M.T @ t`` and rotation matrix
``M.T @ R @ M`` (convert the latter to a MuJoCo wxyz quaternion).
This helper does not add a floating joint. Move an existing parent body or frame
to position the hand in the scene.

Source-wide mass normalization (`settotalmass`) is resolved into per-body
inertias before attachment. The parent scene can still impose its own global
mass scaling when compiled.

Native attachment retains the packaged meshes, joints, actuators, equalities,
tendons and collision definitions. The parent scene's global solver/options win;
MuJoCo can warn when a hand's options differ. No collision shapes or motors are
invented: Barrett and Robotiq packages provide kinematics and visual geometry,
so simulation requires application-appropriate dynamics, collision and actuation
configuration. Their packaged inertias are placeholders inferred from visual
mesh volume at density 1000 kg/m³, rather than the source URDF inertials.
A compiled model alone does not establish grasping performance.

An explicit `model_path` has the same meaning as in `create_model`. See
[hand sources](hand-sources.md) for description provenance and licenses.
