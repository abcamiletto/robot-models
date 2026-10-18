"""Rigid robots of Body Models skins, with shape-dependent geometry and exact skins."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3
from trimesh import Trimesh

from robot_models import _assets as assets
from robot_models import _catalog as catalog
from robot_models._base import GeometryKind, ModelConfig, ParameterSpec, RigidBodyModel
from robot_models._common import eye_as, zeros_as
from robot_models._common import rigid as rigid_ops
from robot_models._runtime import ArrayRuntime
from robot_models._skinned._sources import SOURCES, Source
from robot_models.hands._base import Hand
from robot_models.humanoids._base import Humanoid

Array = Any
Gender = Literal["neutral", "male", "female"]


@dataclass(frozen=True)
class ShapeAssets:
    """Shape directions of a skinned robot's rest geometry, and its pose-coordinate frames.

    Each moving joint (Body Models' pose row ``pose_rows``) rotates by its three hinges, in
    order, about the columns of ``hinge_frames``: the third negated where ``hinge_signs`` is -1,
    so that every frame is a rotation.
    """

    pose_rows: list[int]
    offset_dirs: Float[Array, "S J 3"]
    visual_dirs: Float[Array, "S V 3"]
    collision_dirs: Float[Array, "S C 3"]
    root: Float[Array, "3"]
    root_dirs: Float[Array, "S 3"]
    hinge_frames: Float[Array, "M 3 3"]
    hinge_signs: Float[Array, "M"]


class Skinned(RigidBodyModel):
    """A Body Models skin as a rigid robot: its links cut from the rest skin, three hinges per joint.

    ``shape`` takes Body Models' shape coefficients and moves the joints, links and collision
    pieces as the skin's rest shape moves them. The visual geometry posed by
    :meth:`forward_meshes` is the exact skin from Body Models; :attr:`link_meshes` are its
    rigid links at the mean shape.
    """

    _state_fields = ("_assets", "_shape", "_skin")
    _shape: ShapeAssets
    _skin: Any

    @property
    def _source(self) -> Source:
        return source_of(self.name)

    def _load(self, xml_name: str, model_path: Path | str | None, runtime: ArrayRuntime) -> None:
        directory = assets.model_dir(self.name, model_path)
        variant = Path(xml_name).stem
        if not (directory / xml_name).is_file():
            raise FileNotFoundError(
                f"{self.name} {self._source.variant_key} {variant!r} is not built in {directory}: configure its "
                f"Body Models file with `body-models set {self._source.body_model}-{variant} PATH`."
            )
        super()._load(xml_name, directory, runtime)
        axes = np.array(runtime.to_numpy(self._assets.hinge_axes), np.float32)  # a copy: the frames change
        frames = np.swapaxes(axes.reshape(-1, 3, 3), 1, 2)
        signs = np.sign(np.linalg.det(frames)).astype(np.float32)
        frames[..., 2] *= signs[:, None]
        with np.load(directory / f"{variant}.npz") as data:
            dirs = {key: data[key].astype(np.float32) for key in ("offset_dirs", "visual_dirs", "collision_dirs")}
            root = {key: data[key].astype(np.float32) for key in ("root", "root_dirs")}
            pose_rows = data["pose_rows"].tolist()
        shape = ShapeAssets(pose_rows, **dirs, **root, hinge_frames=frames, hinge_signs=signs)
        self._shape = runtime._materialize(shape)
        self._skin = self._source.create(variant, runtime.name)

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        return {"shape": ParameterSpec((self.num_shapes,), "identity"), **super().parameter_spec}

    @property
    def num_shapes(self) -> int:
        """Number of Body Models shape coefficients."""
        return self._shape.offset_dirs.shape[0]

    @property
    def faces(self) -> Int[Array, "F 3"]:
        """Faces of the skin posed by ``forward_meshes()``."""
        return self._skin.faces

    @property
    def num_vertices(self) -> int:
        return self._skin.num_vertices

    def _forward_skeleton(
        self,
        pose: Float[Array, "*batch Q"],
        global_rotation: Float[Array, "*batch 3"] | None,
        global_translation: Float[Array, "*batch 3"] | None,
        joint_indices: Sequence[int] | None,
        shape: Float[Array, "*batch S"] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        return rigid_ops.forward_skeleton(
            self._assets,
            pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            local_offsets=self._shaped(shape, self._assets.local_offsets, self._shape.offset_dirs),
            xp=self._runtime.xp,
        )

    def forward_meshes(self, *, geometry: GeometryKind = "visual", **parameters: Float[Array, "..."]) -> list[Trimesh]:
        """Pose the exact skin (``"visual"``) or the rigid collision pieces, one mesh per batch element."""
        if geometry == "visual":
            vertices = self._runtime.to_numpy(self._skin.forward_vertices(**self.to_body_models(**parameters)))
            faces = self._runtime.to_numpy(self._skin.faces)
            return [Trimesh(v, faces.copy(), process=False) for v in vertices.reshape(-1, *vertices.shape[-2:])]
        meshes = self._geometry_assets(geometry)
        shape = parameters.get("shape")
        return rigid_ops.forward_meshes_from_links(
            self.forward_links(geometry=geometry, **parameters),
            meshes,
            vertices=self._shaped(shape, meshes.vertices, self._shape.collision_dirs),
            to_numpy=self._runtime.to_numpy,
            xp=self._runtime.xp,
        )

    def _shaped(
        self, shape: Float[Array, "*batch S"] | None, rest: Float[Array, "N 3"], dirs: Float[Array, "S N 3"]
    ) -> Float[Array, "*batch N 3"] | None:
        """Rest positions moved by shape coefficients, or ``None`` for the mean shape."""
        if shape is None:
            return None
        xp = self._runtime.xp
        dirs = xp.asarray(dirs, dtype=shape.dtype)
        moved = shape @ xp.reshape(dirs, (dirs.shape[0], -1))
        return xp.asarray(rest, dtype=shape.dtype) + xp.reshape(moved, (*shape.shape[:-1], *rest.shape))

    def from_body_models(self, **parameters: Float[Array, "..."]) -> dict[str, Float[Array, "..."]]:
        """Convert Body Models parameters (axis-angle, as ``create_model`` defaults) to this model's.

        The hinge angles reproduce each joint's rotation exactly; the root's rotation and
        translation absorb Body Models' global and root rotations, since this model's root
        joint sits at the origin. ``shape`` passes through, defaulting to the mean shape.
        """
        xp = self._runtime.xp
        pose_name, root_name = self._POSE, self._source.root_rotation
        pose = parameters[pose_name]
        batch = tuple(pose.shape[:-2])
        local = SO3.convert(pose[..., self._shape.pose_rows, :], src="axis_angle", dst="rotmat", xp=xp)
        frames = xp.asarray(self._shape.hinge_frames, dtype=pose.dtype)
        m = frames.mT @ local @ frames  # intrinsic rotations about the frame's x, y and z axes
        signs = xp.asarray(self._shape.hinge_signs, dtype=pose.dtype)
        first = xp.arctan2(-m[..., 1, 2], m[..., 2, 2])
        second = xp.arctan2(m[..., 0, 2], xp.sqrt(m[..., 0, 0] ** 2 + m[..., 0, 1] ** 2))
        third = xp.arctan2(-m[..., 0, 1], m[..., 0, 0]) * signs
        angles = xp.stack([first, second, third], axis=-1)
        shape = parameters.get("shape")
        if shape is None:
            shape = zeros_as(pose, shape=(*batch, self.num_shapes), xp=xp)
        global_rot = self._rotation(parameters.get("global_rotation"), pose, batch)
        root_rot = self._rotation(parameters.get(root_name), pose, batch)
        translation = parameters.get("global_translation")
        root = xp.squeeze(global_rot @ self._root(shape)[..., None], axis=-1)
        return {
            pose_name: xp.reshape(angles, (*batch, -1)),
            "global_rotation": SO3.convert(global_rot @ root_rot, src="rotmat", dst="axis_angle", xp=xp),
            "global_translation": root if translation is None else root + translation,
            "shape": shape,
        }

    def to_body_models(self, **parameters: Float[Array, "..."]) -> dict[str, Float[Array, "..."]]:
        """Convert this model's parameters to Body Models' (axis-angle), the root rotation global."""
        xp = self._runtime.xp
        pose = parameters[self._POSE]
        batch = tuple(pose.shape[:-1])
        angles = rigid_ops.hinge_angles(self._assets, pose, xp=xp)
        axes = xp.asarray(self._assets.hinge_axes, dtype=pose.dtype)
        hinges = SO3.convert(angles[..., None], src="hinge", dst="rotmat", src_kwargs={"axes": axes}, xp=xp)
        local = hinges[..., 0::3, :, :] @ hinges[..., 1::3, :, :] @ hinges[..., 2::3, :, :]
        order = np.argsort(self._shape.pose_rows)  # Body Models' joint order
        body_pose = SO3.convert(local[..., order, :, :], src="rotmat", dst="axis_angle", xp=xp)
        shape = parameters.get("shape")
        if shape is None:
            shape = zeros_as(pose, shape=(*batch, self.num_shapes), xp=xp)
        global_rot = self._rotation(parameters.get("global_rotation"), pose, batch)
        translation = parameters.get("global_translation")
        if translation is None:
            translation = zeros_as(pose, shape=(*batch, 3), xp=xp)
        root = xp.squeeze(global_rot @ self._root(shape)[..., None], axis=-1)
        return {
            self._POSE: body_pose,
            self._source.root_rotation: zeros_as(pose, shape=(*batch, 3), xp=xp),
            "global_rotation": SO3.convert(global_rot, src="rotmat", dst="axis_angle", xp=xp),
            "global_translation": translation - root,
            "shape": shape,
        }

    def _root(self, shape: Float[Array, "*batch S"]) -> Float[Array, "*batch 3"]:
        """The root joint's rest position in Body Models' frame."""
        xp = self._runtime.xp
        root = xp.asarray(self._shape.root, dtype=shape.dtype)
        return root + shape @ xp.asarray(self._shape.root_dirs, dtype=shape.dtype)

    def _rotation(self, value: Float[Array, "*batch 3"] | None, like: Array, batch: tuple[int, ...]) -> Array:
        xp = self._runtime.xp
        if value is None:
            return eye_as(zeros_as(like, shape=(*batch, 3), xp=xp), batch_dims=batch, xp=xp)
        return SO3.convert(value, src="axis_angle", dst="rotmat", xp=xp)


def source_of(name: str) -> Source:
    """The Body Models source of a locally built model."""
    body_model = catalog.model_spec(name).body_model
    if body_model is None:
        raise ValueError(f"{name!r} is not built from Body Models")
    return SOURCES[body_model]


class SkinnedHand(Skinned, Hand):
    """A skinned hand posed by ``hand_pose``, with Body Models' ``shape``."""

    def forward_skeleton(
        self,
        hand_pose: Float[Array, "*batch Q"],
        *,
        shape: Float[Array, "*batch S"] | None = None,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed joint transforms of a hand of ``shape`` (the mean shape by default)."""
        return self._forward_skeleton(hand_pose, global_rotation, global_translation, joint_indices, shape)


class SkinnedHumanoid(Skinned, Humanoid):
    """A skinned humanoid posed by ``body_pose``, with Body Models' ``shape``."""

    def __init__(self, name: str, *, gender: Gender = "neutral", model_path: Path | str | None = None, runtime) -> None:
        name = catalog.normalize_name(name)
        if gender not in source_of(name).variants:
            raise ValueError(f"{name!r} supports genders: {source_of(name).variants}")
        self._config = ModelConfig(name, variant=gender)
        self._load(f"{gender}.xml", model_path, runtime)

    @property
    def gender(self) -> Gender:
        return self._config.variant  # type: ignore[return-value]

    def forward_skeleton(
        self,
        body_pose: Float[Array, "*batch Q"],
        *,
        shape: Float[Array, "*batch S"] | None = None,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed joint transforms of a body of ``shape`` (the mean shape by default)."""
        return self._forward_skeleton(body_pose, global_rotation, global_translation, joint_indices, shape)
