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

A model loads one MJCF file, `xml_path`, which MuJoCo reads as it is. Its bodies are the MJCF
bodies (`body_names`, the frames of `forward_skeleton`), its pose coordinates the independent
hinges (`dof_names`, `dof_limits`), and `hinge_names` every hinge, coupled ones included (the
angles of `to_qpos`). `visual` and `collision` are
[mesh sets](collisions.md). A humanoid's `symmetric_bodies` gives left/right body index pairs;
swapping these indices alone does not reflect rotations or mirror a pose.

::: robot_models.RigidBodyModel

::: robot_models.MeshSet

## Metadata

::: robot_models.Dof

::: robot_models.Joint

## Runtimes

::: robot_models.ArrayRuntime
