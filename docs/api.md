# API Reference

## Model creation

::: robot_models.create_model
    options:
      show_source: false

::: robot_models.list_models
    options:
      show_source: false

## Model contract

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
