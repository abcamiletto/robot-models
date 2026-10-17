# API Reference

## Model creation

Choose the backend through the import, as in Body Models:

```python
from robot_models.hands.shadow.torch import ShadowHand
from robot_models.humanoids.g1.torch import G1

hand = ShadowHand(side="right")
humanoid = G1()
```

Constructors take no runtime argument. `list_models(family="hands")` and
`list_models(family="humanoids")` discover canonical definitions. The optional
generic factory below follows Body Models' catalog factory.

::: robot_models.create_model
    options:
      show_source: false

::: robot_models.list_models
    options:
      show_source: false

## MuJoCo attachment

::: robot_models.add_hand
    options:
      show_source: false

## Model contract

Both families expose local `collision_meshes` and posed
`forward_meshes(geometry="collision", **params)`. See
[visual and collision geometry](collisions.md) for examples and source details.

`forward_skeleton(..., joint_indices=...)` returns only the requested joints and
evaluates only their kinematic chains, so a few end effectors cost less than the
full skeleton.

`symmetric_joints` lists `(left, right)` index pairs over the whole native
skeleton, including joints outside `Joint`; unpaired joints lie on the midline.
Use it for symmetry losses or left/right swaps:

```python
order = list(range(model.num_joints))
for left, right in model.symmetric_joints:
    order[left], order[right] = right, left
swapped = model.forward_skeleton(**params)[..., order, :, :]
```

Swapping indices does not mirror a pose: the rotations must also be reflected in
the model's frame and parameterization.

::: robot_models.RigidBodyModel
    options:
      show_source: false

## Metadata

::: robot_models.ParameterSpec
    options:
      show_source: false

::: robot_models.Joint
    options:
      show_source: false

## Runtimes

::: robot_models.ArrayRuntime
    options:
      show_source: false
