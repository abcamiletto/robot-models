"""PSYONIC Ability Hand coupled-joint kinematics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jaxtyping import Float
from nanomanifold import SO3

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
    """Compute world-space Ability Hand joint transforms."""
    if pose.ndim < 1 or pose.shape[-1] != len(actuated_joint_indices):
        raise ValueError(f"Ability Hand pose must have shape [..., {len(actuated_joint_indices)}], got {pose.shape}")
    dtype = pose.dtype
    coeffs = xp.asarray(coupled_polycoef, dtype=dtype)
    driver_pose = pose[..., coupled_driver_indices]
    # MuJoCo joint equality: coupled = c0 + c1 * driver + ... + c4 * driver^4.
    coupled_pose = coeffs[:, 4]
    for power in (3, 2, 1, 0):
        coupled_pose = coupled_pose * driver_pose + coeffs[:, power]

    hinge_pose = xp.concat([pose, coupled_pose], axis=-1)
    axes = xp.concat(
        [xp.asarray(actuated_joint_axes, dtype=dtype), xp.asarray(coupled_joint_axes, dtype=dtype)],
        axis=0,
    )
    rotations = SO3.convert(hinge_pose[..., None], src="hinge", dst="rotmat", src_kwargs={"axes": axes}, xp=xp)
    return rigid.forward_skeleton_from_local_rotations(
        rotations,
        local_offsets=local_offsets,
        rest_local_rotations=rest_local_rotations,
        actuated_joint_indices=[*actuated_joint_indices, *coupled_joint_indices],
        parents=parents,
        global_translation=global_translation,
        global_rotation=global_rotation,
        joint_indices=joint_indices,
        xp=xp,
    )
