"""Shared rigid articulated mesh helpers."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3
from trimesh import Trimesh
from trimesh.util import concatenate

from robot_models._common.kinematics import affine_transforms, select_joint_chains
from robot_models._common.ops import at_set, eye_as, zeros_as

Array = Any
_ToNumpy = Callable[[Any], np.ndarray]


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
    vertices: Float[Array, "V 3"]
    faces: Int[Array, "F 3"]
    link_names: list[str]
    link_joint_indices: list[int]
    link_vertex_starts: list[int]
    link_vertex_counts: list[int]
    link_face_starts: list[int]
    link_face_counts: list[int]


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
    xp: Any,
) -> Float[Array, "... J 4 4"]:
    """Compute world-space joint transforms from pose coordinates."""
    num_dofs = len(assets.actuated_joint_names)
    if pose.ndim < 1 or pose.shape[-1] != num_dofs:
        raise ValueError(f"pose must have shape [..., {num_dofs}], got {tuple(pose.shape)}")
    batch_shape = tuple(pose.shape[:-1])
    num_joints = len(assets.parents)
    axes = xp.asarray(assets.hinge_axes, dtype=pose.dtype)
    angles = hinge_angles(assets, pose, xp=xp)
    hinge_rot = SO3.convert(angles[..., None], src="hinge", dst="rotmat", src_kwargs={"axes": axes}, xp=xp)

    local_rot = eye_as(hinge_rot, batch_dims=(*batch_shape, num_joints), xp=xp)
    local_rot = at_set(local_rot, (..., assets.hinge_joint_indices, slice(None), slice(None)), hinge_rot, xp=xp)
    rest_rot = xp.asarray(assets.rest_local_rotations, dtype=pose.dtype)
    return forward_skeleton_from_local_transforms(
        rest_rot @ local_rot,
        local_offsets=assets.local_offsets,
        parents=assets.parents,
        global_translation=global_translation,
        global_rotation=global_rotation,
        joint_indices=joint_indices,
        xp=xp,
    )


def forward_skeleton_from_local_transforms(
    local_rotations: Float[Array, "... J 3 3"],
    *,
    local_offsets: Float[Array, "J 3"],
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

    local_t = xp.asarray(local_offsets, dtype=dtype)

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
        local_pos = xp.squeeze(parent_rot @ local_t[joint][..., None], axis=-1)
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
    vertices: Float[Array, "V 3"],
    faces: Int[Array, "F 3"],
    link_vertex_starts: list[int],
    link_vertex_counts: list[int],
    link_face_starts: list[int],
    link_face_counts: list[int],
    *,
    to_numpy: _ToNumpy,
    xp: Any,
) -> list[Trimesh]:
    """Build one concatenated ``Trimesh`` per batch element."""
    link_rot = links[..., :3, :3]
    link_pos = links[..., :3, 3]
    source_vertices = xp.asarray(vertices, dtype=links.dtype)
    source_faces = xp.asarray(faces)
    batch_size = _batch_size(links)
    meshes_by_batch: list[list[Trimesh]] = [[] for _ in range(batch_size)]

    for link_idx in range(len(link_vertex_starts)):
        vertex_start = link_vertex_starts[link_idx]
        vertex_count = link_vertex_counts[link_idx]
        face_start = link_face_starts[link_idx]
        face_count = link_face_counts[link_idx]
        local_vertices = source_vertices[vertex_start : vertex_start + vertex_count]
        transformed = xp.squeeze(link_rot[..., link_idx, None, :, :] @ local_vertices[..., None], axis=-1)
        mesh_vertices = transformed + link_pos[..., link_idx, None, :]
        mesh_faces = source_faces[face_start : face_start + face_count] - vertex_start
        batched_vertices = _as_batched_vertices(mesh_vertices, batch_size=batch_size, to_numpy=to_numpy)
        faces_np = to_numpy(mesh_faces)
        for batch_idx, batch_vertices in enumerate(batched_vertices):
            meshes_by_batch[batch_idx].append(_make_trimesh(vertices=batch_vertices, faces=faces_np))

    return [_concatenate_meshes(batch_meshes) for batch_meshes in meshes_by_batch]


def link_meshes(
    vertices: Float[Array, "V 3"],
    faces: Int[Array, "F 3"],
    link_vertex_starts: list[int],
    link_vertex_counts: list[int],
    link_face_starts: list[int],
    link_face_counts: list[int],
    *,
    to_numpy: _ToNumpy,
) -> list[Trimesh]:
    """Build one link-local mesh per packed geometry range."""
    vertices = to_numpy(vertices)
    faces = to_numpy(faces)
    meshes = []
    for vertex_start, vertex_count, face_start, face_count in zip(
        link_vertex_starts,
        link_vertex_counts,
        link_face_starts,
        link_face_counts,
        strict=True,
    ):
        link_vertices = vertices[vertex_start : vertex_start + vertex_count]
        link_faces = faces[face_start : face_start + face_count] - vertex_start
        meshes.append(_make_trimesh(vertices=link_vertices, faces=link_faces))
    return meshes


def _make_trimesh(
    *,
    vertices: Float[np.ndarray, "V 3"],
    faces: Int[np.ndarray, "F 3"],
) -> Trimesh:
    return Trimesh(vertices=vertices, faces=faces, process=False)


def _as_batched_vertices(
    vertices: Float[Array, "... V 3"],
    *,
    batch_size: int,
    to_numpy: _ToNumpy,
) -> Float[np.ndarray, "B V 3"]:
    vertices = to_numpy(vertices)
    if vertices.ndim == 2:
        vertices = vertices[None]
    if vertices.ndim < 3:
        raise ValueError("forward_meshes expects vertices with shape [..., V, 3].")
    return vertices.reshape(batch_size, vertices.shape[-2], vertices.shape[-1])


def _batch_size(links: Float[Array, "... L 4 4"]) -> int:
    batch_shape = links.shape[:-3]
    if not batch_shape:
        return 1
    return int(np.prod(batch_shape))


def _concatenate_meshes(meshes: list[Trimesh]) -> Trimesh:
    if not meshes:
        return Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64), process=False)
    return concatenate(meshes)
