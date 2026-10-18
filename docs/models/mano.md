# MANO

`ManoRobot` is MANO (Romero et al., SIGGRAPH Asia 2017) as a rigid robot hand:
Body Models' `create_model("mano", side=side, flat_hand_mean=True)`, the left model
with `flip_shapedirs=True` so that a shape means the same hand on either side.
It has 16 skeleton joints, MANO's, and 45 `hand_pose` coordinates: three hinges
per finger joint, about its anatomical bend, spread and twist axes.

```python
from robot_models.hands.mano.numpy import ManoRobot

hand = ManoRobot(side="right")
params = hand.get_rest_pose(hands="rest")
params["shape"] = betas  # Body Models' 10 shape coefficients; the mean shape by default
skeleton = hand.forward_skeleton(**params)
skin = hand.forward_meshes(**params)  # MANO's skin, exactly
pieces = hand.forward_meshes(geometry="collision", **params)
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

## Kinematics

At any pose and shape, the joint frames are Body Models' `forward_skeleton`.
Convert Body Models parameters (axis-angle `hand_pose` 15×3, `wrist_rotation`,
`global_rotation`, `global_translation`, `shape`) with `from_body_models`, and back
with `to_body_models`:

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

Joint ranges are manotorch's anatomical limits, recorded for `close`, presets and
validation but not enforced in MuJoCo: MANO data often leaves them. Fingers' twist,
and the PIP and DIP spread, have empty ranges.

## Geometry

The links are the flat, rest skin cut by joint: each face goes to the joint that
skins its corners most, and each opening is closed by a fan to its centroid.
At rest they are the skin. Posed, they move rigidly, so they depart from MANO's skin
where it blends between joints and where its pose correctives act, by about 1 cm
at the palm and thumb base in a relaxed grasp; `forward_meshes()` returns the
exact skin from Body Models, `link_meshes` the rigid links at the mean shape.

The collision geometry is a CoACD convex decomposition (Wei et al., SIGGRAPH 2022)
of the right mean-shape links, about 380 pieces; the left hand's are mirrored.
Every rest quantity is linear in the shape coefficients: joint offsets, link and
piece vertices follow the rest skin, pieces by inverse-distance weights over
their link's four nearest vertices. Links have 1100 kg/m³ tissue density.

## Control

`dofs` mark bend as closing flexion, spread as abduction and twist as rotation.
`effort_limits` are nominal human maximal voluntary torques of each bend and
spread hinge (the thumb's base: also twist), from fingertip strength times each
joint's lever; the hinges no muscle turns are unbounded, as the joint's passive
structure holds them. Presets: `flat` and `open` are the flat hand, `rest` a
relaxed curl, `closed` every bend at its upper limit.

```python
import mujoco
from robot_models import add_hand

scene = mujoco.MjSpec()
mount, root = add_hand(scene, "mano", side="left", free=True, shape=betas)
```

The packaged MJCF has a position actuator per hinge (kp 10 N·m/rad), and excludes
contacts between links that touch at rest.

## API

::: robot_models.hands.mano.numpy.ManoRobot
