# TALOS

TALOS is a rigid articulated PAL Robotics TALOS model with STL link meshes
attached to a 48-joint render skeleton. `body_pose` has 44 hinge coordinates:
torso, head, two 7-DoF arms, two 7-joint grippers, and two 6-DoF legs.

## Setup

TALOS downloads automatically on first use from the
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository, which records the original MuJoCo Menagerie
`pal_talos` provenance (Apache-2.0). To prefetch the assets:

```bash
robot-models download talos
```

When passed manually, `model_path` should point to `talos.xml` or to a
directory that contains `talos.xml` and `meshes/`.

## Notes

TALOS does not define `skin_weights`. Use `forward_links()` for link transforms
and `forward_meshes()` for renderable meshes.

Skeleton joint names move the side to the front of the MJCF joint name, so
`arm_left_1_joint` becomes `left_arm_1_skel`. The `head`, `left_foot`, and
`right_foot` joints are leaf joints at the MJCF sites with the same names.

Each gripper exposes all seven MJCF hinge joints as independent coordinates.
The MJCF equality constraints that couple the gripper joints are not applied
by the kinematics.

## API

::: robot_models.talos.numpy.Talos
