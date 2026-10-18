# MANO

`ManoRobot` is MANO (Romero et al., SIGGRAPH Asia 2017) as a rigid robot hand:
Body Models' `create_model("mano", side=side, flat_hand_mean=True)`, the left model
with `flip_shapedirs=True` so that a shape means the same hand on either side.
It has 16 skeleton joints, MANO's, and 45 `hand_pose` coordinates: three hinges
per finger joint, about its anatomical bend, spread and twist axes.

```python
from robot_models.hands.mano.numpy import ManoRobot

hand = ManoRobot(side="right", shape=betas)  # Body Models' 10 coefficients; the mean by default
params = hand.get_rest_pose(hands="rest")
skeleton = hand.forward_skeleton(**params)
links = hand.forward_meshes(**params)  # rigid links
pieces = hand.forward_meshes(geometry="collision", **params)
skin = hand.forward_vertices(**params)  # MANO's skin, exactly; faces in hand.skin_faces
```

## Assets are built locally

MANO's model files are under the [MANO license](https://mano.is.tue.mpg.de/license.html)
and cannot be redistributed, so nothing is downloaded. On first use, robot-models
builds the assets from your own files, as Body Models resolves them
(`body-models set mano-right /path/to/MANO_RIGHT.pkl`, likewise `mano-left`), into
the asset cache, or into a configured `model_path`. Building needs
`robot-models[body-models]` and takes about half a minute; a changed model file
triggers a rebuild. `robot-models download mano` builds ahead of time. Never
publish the built directory: it is derived from the MANO files.

## Shape and geometry pose

A `ManoRobot` is one hand, fixed when it is built: its links and collision pieces are
MANO's skin at `shape` (the mean by default), posed by `conform_to` (this model's
`hand_pose`, flat by default) and cut per joint in its joint frames. The joint offsets
are those of `shape`; `hand.shape` returns it. `add_hand` takes the same arguments.

```python
hand = ManoRobot(side="right", shape=betas, conform_to=grasp_pose)
add_hand(scene, "mano", side="right", shape=betas, conform_to=grasp_pose, free=True)
```

Posed at `conform_to`, the links are the skin and the pieces follow it; elsewhere they
move rigidly. A rigid hand cut at the flat pose departs from the skin by up to about
2 cm at the palm and thumb base in a grasp, which matters for simulation: on 500 MANO
grasps, its pieces touched the object about 12% less than pieces conformed to each
grasp, whose overlap with the object stayed within 1 mm of the skin's. So build one
model per grasp, conformed to it, to simulate grasps; a model loads in a few
milliseconds.

## Kinematics

At any pose, the joint frames are Body Models' `forward_skeleton` at the model's
shape. Convert Body Models parameters (axis-angle `hand_pose` 15×3,
`wrist_rotation`, `global_rotation`, `global_translation`) with `from_body_models`,
and back with `to_body_models`, which also returns the model's `shape`. A `shape`
given to `from_body_models` must be the model's:

```python
params = hand.from_body_models(**mano_params)
mano_params = hand.to_body_models(**params)  # wrist rotation folded into the global one
```

The root joint, the wrist, sits at the origin: `global_rotation` and
`global_translation` place the wrist, unlike Body Models' rotation about the MANO
origin. Each joint rotates by its hinges in order, bend, spread and twist, about
manotorch's anatomical frames of the flat template (Yang et al., CPF, ICCV 2021):
twist runs along the bone, bend is the cross product of twist and an up direction,
spread completes the frame. The three intrinsic Euler angles reach every MANO pose
exactly. The left hand's axes are the right's mirrored, so an angle means the same
motion on either side; positive bend flexes.

`actuated_joint_limits` are enforced, as for every hand, by the MJCF too: human maximal
ranges widened generously, beyond any real hand. 99.9% of 237k synthesized MANO grasps
from six methods lie within; the others bend a joint where no hand can, such as a finger
folded back flat. `anatomical_limits` are manotorch's natural ranges, narrower and
partly empty (a finger's twist, the PIP and DIP spread): a pose outside them is possible
but unnatural, so measure against them how far a grasp lies from a natural hand. `close`
keeps within the enforced limits; the presets are natural poses, within both.

## Geometry

Each skin face goes to the joint that skins its corners most, and each link is closed
by a fan over its openings. `forward_meshes()` poses the rigid links, as for every
robot, and `link_meshes` holds them. `forward_vertices()` poses the exact skin through
Body Models at any pose, as Body Models' own `forward_vertices` does (differentiable,
with `vertex_indices`); its faces are `skin_faces`.

The collision geometry is a CoACD convex decomposition (Wei et al., SIGGRAPH 2022)
of the flat, mean-shape right links, about 380 pieces; the left hand's are mirrored.
At another shape or `conform_to`, each piece vertex follows its link by inverse-distance
weights over the link's four nearest vertices, as grasp-bench does. Links have
1100 kg/m³ tissue density.

## Control

`dofs` mark bend as closing flexion, spread as abduction and twist as rotation.
`effort_limits` are nominal human maximal voluntary torques of each bend and
spread hinge (the thumb's base: also twist), from fingertip strength times each
joint's lever; the hinges no muscle turns are unbounded, as the joint's passive
structure holds them. Presets: `flat` and `open` are the flat hand, `rest` a
relaxed curl, `closed` every bend at its upper anatomical limit.

```python
import mujoco
from robot_models import add_hand

scene = mujoco.MjSpec()
mount, root = add_hand(scene, "mano", side="left", free=True, shape=betas)  # as ManoRobot(side="left", shape=betas)
```

The packaged MJCF has a position actuator per hinge (kp 10 N·m/rad), and excludes
contacts between links that touch at rest.

## API

::: robot_models.hands.mano.numpy.ManoRobot
