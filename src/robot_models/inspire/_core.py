"""Inspire hand coupled-joint kinematics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jaxtyping import Float
from nanomanifold import SO3

from robot_models import _common as common
from robot_models._common import rigid

Array = Any


def forward_skeleton(
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    actuated_joint_axes: Float[Array, "Q 3"],
    actuated_joint_indices: list[int],
    coupled_joint_axes: Float[Array, "C 3"],
    coupled_joint_indices: list[int],
    coupled_driver_indices: list[int],
    coupled_polycoef: Float[Array, "C 5"],
    parents: list[int],
    pose: Float[Array, "B Q"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B 3"] | None = None,
    joint_indices: Sequence[int] | None = None,
    xp: Any,
) -> Float[Array, "B J 4 4"]:
    """Compute world-space Inspire hand joint transforms."""
    if pose.ndim < 1 or pose.shape[-1] != len(actuated_joint_indices):
        raise ValueError(f"Inspire pose must have shape [..., {len(actuated_joint_indices)}], got {tuple(pose.shape)}")
    dtype = pose.dtype
    # Coupled angles follow MuJoCo joint equalities: polycoef[0] + polycoef[1] * q + ... + polycoef[4] * q^4.
    coeffs = xp.asarray(coupled_polycoef, dtype=dtype)
    driver_pose = pose[..., coupled_driver_indices]
    coupled_pose = coeffs[:, -1]
    for i in range(coeffs.shape[-1] - 2, -1, -1):
        coupled_pose = coupled_pose * driver_pose + coeffs[:, i]

    angles = xp.concat([pose, coupled_pose], axis=-1)
    axes = xp.concat([xp.asarray(actuated_joint_axes, dtype=dtype), xp.asarray(coupled_joint_axes, dtype=dtype)])
    hinge_rot = SO3.convert(angles[..., None], src="hinge", dst="rotmat", src_kwargs={"axes": axes}, xp=xp)

    batch_shape = tuple(pose.shape[:-1])
    num_joints = len(parents)
    local_rot = common.eye_as(hinge_rot, batch_dims=(*batch_shape, num_joints), xp=xp)
    hinge_indices = [*actuated_joint_indices, *coupled_joint_indices]
    local_rot = common.at_set(local_rot, (..., hinge_indices, slice(None), slice(None)), hinge_rot, xp=xp)
    rest_rot = xp.asarray(rest_local_rotations, dtype=dtype)
    local_rot = xp.broadcast_to(rest_rot, (*batch_shape, num_joints, 3, 3)) @ local_rot
    return rigid.forward_skeleton_from_local_transforms(
        local_rot,
        local_offsets=local_offsets,
        parents=parents,
        global_translation=global_translation,
        global_rotation=global_rotation,
        joint_indices=joint_indices,
        xp=xp,
    )
