# API Reference

## Model creation

Choose the array backend through the import:

```python
from robot_models.hands.shadow.torch import ShadowHand
from robot_models.humanoids.g1.torch import G1

hand = ShadowHand(side="right")
humanoid = G1()
```

Use `.numpy` or `.jax` for the other backends. `list_models(family="hands")`
and `list_models(family="humanoids")` list models; `create_model` is the generic factory.

::: robot_models.create_model

::: robot_models.list_models

## MuJoCo attachment

::: robot_models.add_hand

## Model contract

Both families provide local `collision_meshes` and posed
`forward_meshes(geometry="collision", **params)`; see [geometry](collisions.md).

`forward_skeleton(..., joint_indices=...)` evaluates only the requested joint
chains. `symmetric_joints` gives left/right index pairs across the full skeleton;
swapping these indices alone does not reflect rotations or mirror a pose.

::: robot_models.RigidBodyModel

## Metadata

::: robot_models.ParameterSpec

::: robot_models.Joint

## Runtimes

::: robot_models.ArrayRuntime
