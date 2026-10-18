"""Shared rigid articulated mesh helpers."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3
from trimesh import Trimesh

from robot_models._common.kinematics import affine_transforms, select_joint_chains
from robot_models._common.ops import at_set, eye_as, zeros_as

Array = Any
_ToNumpy = Callable[[Any], np.ndarray]


@dataclass(frozen=True)
class MeshAssets:
    """Packed meshes, each expressed in its skeleton joint's local frame."""

    vertices: Float[Array, "V 3"]
    faces: Int[Array, "F 3"]
    names: list[str]
    joint_indices: list[int]
    vertex_offsets: list[int]
    face_offsets: list[int]


@dataclass(frozen=True)
class RigidAssets:
    """Skeleton, hinge, and link-mesh data of one rigid articulated model.

    Every hinge angle is a polynomial of one pose coordinate:
    ``angle[h] = sum_k hinge_polycoef[h, k] * pose[hinge_drivers[h]] ** k``.
    Independent hinges use the identity polynomial on their own coordinate.
    """

    joint_names: list[str]
    parents: list[int]
    local_offsets: Float[Array, "J 3"]
    rest_local_rotations: Float[Array, "J 3 3"]
    hinge_joint_indices: list[int]
    hinge_axes: Float[Array, "H 3"]
    hinge_drivers: list[int]
    hinge_polycoef: Float[Array, "H 5"]
    actuated_joint_names: list[str]
    actuated_joint_limits: Float[Array, "Q 2"]
    visual: MeshAssets
    collision: MeshAssets


def hinge_angles(assets: RigidAssets, pose: Float[Array, "... Q"], *, xp: Any) -> Float[Array, "... H"]:
    """Evaluate every hinge angle from the pose coordinates."""
    coef = xp.asarray(assets.hinge_polycoef, dtype=pose.dtype)
    x = pose[..., assets.hinge_drivers]
    return coef[:, 0] + x * (coef[:, 1] + x * (coef[:, 2] + x * (coef[:, 3] + x * coef[:, 4])))


def forward_skeleton(
    assets: RigidAssets,
    pose: Float[Array, "... Q"],
    *,
    global_rotation: Float[Array, "... 3"] | None,
    global_translation: Float[Array, "... 3"] | None,
    joint_indices: Sequence[int] | None,
    local_offsets: Float[Array, "... J 3"] | None = None,
    xp: Any,
) -> Float[Array, "... J 4 4"]:
    """Compute world-space joint transforms from pose coordinates.

    A body's hinges rotate in MJCF order, each about its axis after the previous ones, as in MuJoCo.
    ``local_offsets`` replaces the asset offsets, for instance with batched shape-dependent ones.
    """
    num_dofs = len(assets.actuated_joint_names)
    if pose.ndim < 1 or pose.shape[-1] != num_dofs:
        raise ValueError(f"pose must have shape [..., {num_dofs}], got {tuple(pose.shape)}")
    batch_shape = tuple(pose.shape[:-1])
    num_joints = len(assets.parents)
    axes = xp.asarray(assets.hinge_axes, dtype=pose.dtype)
    angles = hinge_angles(assets, pose, xp=xp)
    hinge_rot = SO3.convert(angles[..., None], src="hinge", dst="rotmat", src_kwargs={"axes": axes}, xp=xp)

    local_rot = xp.asarray(assets.rest_local_rotations, dtype=pose.dtype)
    identity = eye_as(hinge_rot, batch_dims=(*batch_shape, num_joints), xp=xp)
    for hinges, joints in _hinge_slots(assets.hinge_joint_indices):
        slot = at_set(identity, (..., joints, slice(None), slice(None)), hinge_rot[..., hinges, :, :], xp=xp)
        local_rot = local_rot @ slot
    return forward_skeleton_from_local_transforms(
        local_rot,
        local_offsets=assets.local_offsets if local_offsets is None else local_offsets,
        parents=assets.parents,
        global_translation=global_translation,
        global_rotation=global_rotation,
        joint_indices=joint_indices,
        xp=xp,
    )


def _hinge_slots(hinge_joints: Sequence[int]) -> list[tuple[list[int], list[int]]]:
    """Group hinges by their rank within their body: the k-th group holds every body's k-th hinge."""
    ranks: dict[int, int] = {}
    slots: list[tuple[list[int], list[int]]] = []
    for hinge, joint in enumerate(hinge_joints):
        rank = ranks[joint] = ranks.get(joint, -1) + 1
        if rank == len(slots):
            slots.append(([], []))
        slots[rank][0].append(hinge)
        slots[rank][1].append(joint)
    return slots


def forward_skeleton_from_local_transforms(
    local_rotations: Float[Array, "... J 3 3"],
    *,
    local_offsets: Float[Array, "*batch J 3"],
    parents: list[int],
    global_translation: Float[Array, "... 3"] | None = None,
    global_rotation: Float[Array, "... 3"] | None = None,
    joint_indices: Sequence[int] | None = None,
    xp: Any,
) -> Float[Array, "... J 4 4"]:
    """Compute rigid hierarchy transforms from local joint transforms.

    With ``joint_indices``, only the selected joints and their ancestors are evaluated.
    """
    batch_shape = tuple(local_rotations.shape[:-3])
    dtype = local_rotations.dtype
    outputs = range(len(parents)) if joint_indices is None else tuple(int(joint) for joint in joint_indices)
    chains = select_joint_chains(parents, outputs)
    if not outputs:
        return zeros_as(local_rotations, shape=(*batch_shape, 0, 4, 4), xp=xp)
    if global_translation is None:
        global_translation = zeros_as(local_rotations, shape=(*batch_shape, 3), xp=xp)

    # Offsets of the right dtype pass as they are, keeping their gradients (torch.asarray drops them).
    same = getattr(local_offsets, "dtype", None) == dtype
    local_t = local_offsets if same else xp.asarray(local_offsets, dtype=dtype)

    rot_world: dict[int, Float[Array, "*batch 3 3"]] = {}
    pos_world: dict[int, Float[Array, "*batch 3"]] = {}
    for joint in chains:
        parent = parents[joint]
        if parent < 0:
            rot_world[joint] = local_rotations[..., joint, :, :]
            pos_world[joint] = zeros_as(local_rotations, shape=(*batch_shape, 3), xp=xp)
            continue
        parent_rot = rot_world[parent]
        rot_world[joint] = parent_rot @ local_rotations[..., joint, :, :]
        local_pos = xp.squeeze(parent_rot @ local_t[..., joint, :, None], axis=-1)
        pos_world[joint] = pos_world[parent] + local_pos

    rot = xp.stack([rot_world[joint] for joint in outputs], axis=-3)
    trans = xp.stack([pos_world[joint] for joint in outputs], axis=-2)
    if global_rotation is not None:
        global_rot = SO3.convert(global_rotation, src="axis_angle", dst="rotmat", xp=xp)
        rot = global_rot[..., None, :, :] @ rot
        trans = xp.squeeze(global_rot[..., None, :, :] @ trans[..., None], axis=-1)
    trans = trans + global_translation[..., None, :]
    return affine_transforms(rot, trans, xp=xp)


def forward_meshes_from_links(
    links: Float[Array, "... L 4 4"],
    assets: MeshAssets,
    *,
    vertices: Float[Array, "... V 3"] | None = None,
    to_numpy: _ToNumpy,
    xp: Any,
) -> list[Trimesh]:
    """Build one concatenated ``Trimesh`` per batch element.

    ``vertices`` replaces the link-local asset vertices, for instance with batched shape-dependent ones.
    """
    link_rot = links[..., :3, :3]
    link_pos = links[..., :3, 3]
    source_vertices = xp.asarray(assets.vertices if vertices is None else vertices, dtype=links.dtype)
    chunks = []
    for index, (start, end) in enumerate(zip(assets.vertex_offsets[:-1], assets.vertex_offsets[1:], strict=True)):
        rotated = source_vertices[..., start:end, :] @ link_rot[..., index, :, :].mT
        chunks.append(rotated + link_pos[..., index, None, :])
    vertices = xp.concat(chunks, axis=-2) if chunks else zeros_as(links, shape=(*links.shape[:-3], 0, 3), xp=xp)
    batched = to_numpy(vertices).reshape(prod(links.shape[:-3]), assets.vertex_offsets[-1], 3)
    faces = to_numpy(assets.faces)
    return [Trimesh(vertices=batch_vertices, faces=faces.copy(), process=False) for batch_vertices in batched]


def link_meshes(
    assets: MeshAssets,
    *,
    to_numpy: _ToNumpy,
) -> list[Trimesh]:
    """Build one link-local mesh per packed geometry range."""
    vertices = to_numpy(assets.vertices)
    faces = to_numpy(assets.faces)
    meshes = []
    for vertex_start, vertex_end, face_start, face_end in zip(
        assets.vertex_offsets[:-1],
        assets.vertex_offsets[1:],
        assets.face_offsets[:-1],
        assets.face_offsets[1:],
        strict=True,
    ):
        link_vertices = vertices[vertex_start:vertex_end].copy()
        link_faces = faces[face_start:face_end] - vertex_start
        meshes.append(Trimesh(vertices=link_vertices, faces=link_faces, process=False))
    return meshes
