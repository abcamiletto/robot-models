"""Shared rigid articulated mesh helpers."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3
from trimesh import Trimesh

from robot_models._common.kinematics import affine_transforms
from robot_models._common.ops import at_set, eye_as, zeros_as

Array = Any


@dataclass(frozen=True)
class MeshAssets:
    """Packed meshes, each expressed in its body's frame."""

    vertices: Float[Array, "V 3"]
    faces: Int[Array, "F 3"]
    names: list[str]
    body_indices: list[int]
    vertex_offsets: list[int]
    face_offsets: list[int]


@dataclass(frozen=True)
class RigidAssets:
    """Bodies, hinges and meshes of one rigid articulated model.

    Every hinge angle is a polynomial of one pose coordinate (dof):
    ``angle[h] = sum_k hinge_polycoef[h, k] * pose[hinge_drivers[h]] ** k``.
    Independent hinges, the dofs, use the identity polynomial on their own coordinate.
    """

    body_names: list[str]
    parents: list[int]
    local_offsets: Float[Array, "B 3"]
    rest_local_rotations: Float[Array, "B 3 3"]
    hinge_names: list[str]
    hinge_bodies: list[int]
    hinge_axes: Float[Array, "H 3"]
    hinge_drivers: list[int]
    hinge_polycoef: Float[Array, "H 5"]
    dof_names: list[str]
    dof_limits: Float[Array, "Q 2"]
    dof_efforts: Float[Array, "Q"]
    visual: MeshAssets
    collision: MeshAssets


class MeshSet:
    """A model's visual or collision meshes, each rigid in its body's frame (Y-up, meters).

    ``names`` are the MJCF geom names (or ``{body}_{kind}_{index}`` for unnamed geoms) and
    ``body_indices`` index :attr:`~robot_models.RigidBodyModel.body_names`. Collision meshes
    triangulate curved primitives; the MJCF and ``add_hand`` keep them analytic, and MuJoCo
    collides the convex hull of each mesh.
    """

    def __init__(self, assets: MeshAssets, runtime: Any) -> None:
        self._assets = assets
        self._runtime = runtime

    @property
    def names(self) -> list[str]:
        return list(self._assets.names)

    @property
    def body_indices(self) -> list[int]:
        return list(self._assets.body_indices)

    @property
    def meshes(self) -> list[Trimesh]:
        """One independent mesh per geom, in its body's frame."""
        return [Trimesh(vertices, faces, process=False) for vertices, faces in self.arrays]

    @property
    def arrays(self) -> list[tuple[Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"]]]:
        """Each geom's vertices and faces as NumPy arrays, in its body's frame: :attr:`meshes` without a Trimesh each,
        which takes most of the time for hundreds of collision pieces."""
        vertices = np.array(self._runtime.to_numpy(self._assets.vertices), np.float64)  # a copy: Torch shares buffers
        faces = np.array(self._runtime.to_numpy(self._assets.faces))
        v, f = self._assets.vertex_offsets, self._assets.face_offsets
        return [(vertices[v[i] : v[i + 1]], faces[f[i] : f[i + 1]] - v[i]) for i in range(len(v) - 1)]

    def pose(self, frames: Float[Array, "*batch B 4 4"]) -> list[Trimesh]:
        """One mesh per batch element of body frames (``forward_skeleton``'s): every geom posed, concatenated."""
        xp = self._runtime.xp
        assets = self._assets
        links = frames[..., assets.body_indices, :, :]
        vertices = xp.asarray(assets.vertices, dtype=links.dtype)
        chunks = []
        for index, (start, end) in enumerate(zip(assets.vertex_offsets[:-1], assets.vertex_offsets[1:], strict=True)):
            rotated = vertices[start:end] @ links[..., index, :3, :3].mT
            chunks.append(rotated + links[..., index, None, :3, 3])
        posed = xp.concat(chunks, axis=-2) if chunks else zeros_as(links, shape=(*links.shape[:-3], 0, 3), xp=xp)
        batched = self._runtime.to_numpy(posed).reshape(prod(links.shape[:-3]), len(vertices), 3)
        faces = self._runtime.to_numpy(assets.faces)
        return [Trimesh(vertices=batch, faces=faces.copy(), process=False) for batch in batched]


def hinge_angles(assets: RigidAssets, pose: Float[Array, "... Q"], *, xp: Any) -> Float[Array, "... H"]:
    """Evaluate every hinge angle from the pose coordinates."""
    coef = xp.asarray(assets.hinge_polycoef, dtype=pose.dtype)
    x = pose[..., assets.hinge_drivers]
    return coef[:, 0] + x * (coef[:, 1] + x * (coef[:, 2] + x * (coef[:, 3] + x * coef[:, 4])))


def forward_skeleton(
    assets: RigidAssets,
    pose: Float[Array, "... Q"],
    global_rotation: Float[Array, "... 3"],
    global_translation: Float[Array, "... 3"],
    *,
    xp: Any,
) -> Float[Array, "... B 4 4"]:
    """World transforms of every body, from pose coordinates and the root's axis-angle rotation and translation.

    A body's hinges rotate in MJCF order, each about its axis after the previous ones, as in MuJoCo.
    """
    batch_shape = tuple(pose.shape[:-1])
    axes = xp.asarray(assets.hinge_axes, dtype=pose.dtype)
    angles = hinge_angles(assets, pose, xp=xp)
    hinge_rot = SO3.convert(angles[..., None], src="hinge", dst="rotmat", src_kwargs={"axes": axes}, xp=xp)
    local_rot = xp.asarray(assets.rest_local_rotations, dtype=pose.dtype)
    identity = eye_as(hinge_rot, batch_dims=(*batch_shape, len(assets.parents)), xp=xp)
    for hinges, bodies in _hinge_slots(assets.hinge_bodies):
        local_rot = local_rot @ at_set(
            identity, (..., bodies, slice(None), slice(None)), hinge_rot[..., hinges, :, :], xp=xp
        )

    offsets = xp.asarray(assets.local_offsets, dtype=pose.dtype)
    rotations: list[Any] = []
    positions: list[Any] = []
    for body, parent in enumerate(assets.parents):
        if parent < 0:
            rotations.append(local_rot[..., body, :, :])
            positions.append(zeros_as(pose, shape=(*batch_shape, 3), xp=xp))
            continue
        rotations.append(rotations[parent] @ local_rot[..., body, :, :])
        positions.append(positions[parent] + xp.squeeze(rotations[parent] @ offsets[body][..., None], axis=-1))
    root_rot = SO3.convert(global_rotation, src="axis_angle", dst="rotmat", xp=xp)[..., None, :, :]
    translations = xp.squeeze(root_rot @ xp.stack(positions, axis=-2)[..., None], axis=-1)
    return affine_transforms(
        root_rot @ xp.stack(rotations, axis=-3), translations + global_translation[..., None, :], xp=xp
    )


def _hinge_slots(hinge_bodies: Sequence[int]) -> list[tuple[list[int], list[int]]]:
    """Group hinges by their rank within their body: the k-th group holds every body's k-th hinge."""
    ranks: dict[int, int] = {}
    slots: list[tuple[list[int], list[int]]] = []
    for hinge, body in enumerate(hinge_bodies):
        rank = ranks[body] = ranks.get(body, -1) + 1
        if rank == len(slots):
            slots.append(([], []))
        slots[rank][0].append(hinge)
        slots[rank][1].append(body)
    return slots
