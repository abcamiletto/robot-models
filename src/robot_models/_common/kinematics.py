"""Shared skeleton utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from jaxtyping import Float

from robot_models._common import ops

Array = Any


def affine_transforms(
    linear: Float[Array, "*batch 3 3"],
    translation: Float[Array, "*batch 3"] | None = None,
    *,
    xp: Any,
) -> Float[Array, "*batch 4 4"]:
    """Assemble homogeneous transforms from linear maps and translations."""
    if translation is None:
        translation = ops.zeros_as(linear, shape=(*linear.shape[:-2], 3), xp=xp)

    batch_shape = np.broadcast_shapes(linear.shape[:-2], translation.shape[:-1])
    linear = xp.broadcast_to(linear, (*batch_shape, 3, 3))
    translation = xp.broadcast_to(translation, (*batch_shape, 3))
    upper = xp.concat([linear, translation[..., None]], axis=-1)
    bottom = ops.zeros_as(upper, shape=(*batch_shape, 1, 4), xp=xp)
    bottom = ops.at_set(bottom, (..., 0, 3), 1.0, xp=xp)
    return xp.concat([upper, bottom], axis=-2)


def select_joint_chains(parents: Sequence[int], joint_indices: Sequence[int]) -> tuple[int, ...]:
    """Return ``joint_indices`` and all their ancestors in ascending index order.

    Parents must precede their children, so the result is a valid traversal order.
    """
    num_joints = len(parents)
    if any(joint < 0 or joint >= num_joints for joint in joint_indices):
        raise IndexError(f"joint_indices must be in [0, {num_joints})")

    required: set[int] = set()
    pending = list(joint_indices)
    while pending:
        joint = pending.pop()
        if joint in required:
            continue
        required.add(joint)
        if parents[joint] >= 0:
            pending.append(parents[joint])
    return tuple(sorted(required))
