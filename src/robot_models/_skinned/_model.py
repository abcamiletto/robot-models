"""Rigid robots of Body Models skins, of a shape and geometry fixed at construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Self, cast

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from robot_models import _catalog as catalog
from robot_models._base import RigidBodyModel
from robot_models._common import rigid as rigid_ops
from robot_models._common import zeros_as
from robot_models._skinned._geometry import geometry
from robot_models._skinned._sources import Source
from robot_models.hands._base import Hand
from robot_models.humanoids._base import Humanoid

Array = Any
Gender = Literal["neutral", "male", "female"]


@dataclass(frozen=True)
class Conversion:
    """How this model's parameters map to Body Models'.

    Each moving joint (Body Models' pose row ``pose_rows``) rotates by its three hinges, in
    order, about the columns of ``hinge_frames``: the third negated where ``hinge_signs`` is -1,
    so that every frame is a rotation. ``root`` is the root joint's rest position in Body Models'
    frame, at the model's ``shape``.
    """

    pose_rows: list[int]
    hinge_frames: Float[Array, "M 3 3"]
    hinge_signs: Float[Array, "M"]
    root: Float[Array, "3"]
    shape: Float[Array, "S"]


class Skinned(RigidBodyModel):
    """A Body Models skin as a rigid robot, three hinges per joint, of a shape and geometry fixed at construction.

    Its links are the skin at ``shape`` (the mean by default), posed by ``conform_to`` (this model's
    pose, flat by default), cut per joint in its body frames; its collision pieces follow them. Posed
    at ``conform_to``, the visual links are the skin. :meth:`forward_vertices` poses the exact skin
    of Body Models at any pose, whose faces are :attr:`skin_faces`.
    """

    _state_fields = ("_assets", "_conversion", "_skin")
    _conversion: Conversion
    _skin: Any

    @property
    def _source(self) -> Source:
        return cast(Source, catalog.model_spec(self.name).source)

    def _load(self) -> None:
        path, source = self.xml_path, self._source
        if not path.is_file():
            raise FileNotFoundError(
                f"{self.name} {source.variant_key} {path.stem!r} is not built in {path.parent}: configure its "
                f"Body Models file with `body-models set {source.body_model}-{path.stem} PATH`."
            )
        built = geometry(self.name, path, self._config.shape, self._config.conform_to)
        frames = np.swapaxes(built.assets.hinge_axes.reshape(-1, 3, 3), 1, 2).copy()
        signs = np.sign(np.linalg.det(frames)).astype(np.float32)
        frames[..., 2] *= signs[:, None]
        conversion = Conversion(built.pose_rows, frames, signs, built.root, built.shape)
        assets = built.assets
        if source.efforts:
            assets = replace(assets, dof_efforts=np.asarray([source.efforts[n] for n in assets.dof_names], np.float32))
        self._assets = self._runtime._materialize(assets)
        self._conversion = self._runtime._materialize(conversion)
        self._skin = source.create(path.stem, self._runtime.name)

    @classmethod
    def conformed_to(cls, **arguments: Any) -> tuple[Self, dict[str, Float[Array, "..."]]]:
        """A model conformed to one pose given as Body Models parameters, and that pose as its parameters.

        ``arguments`` are the constructor's (``side`` or ``gender``, ``model_path``) and one pose's Body Models
        parameters (MANO: ``hand_pose``, ``shape``, ``wrist_rotation``, ``global_rotation``,
        ``global_translation``). The model has the parameters' shape and its links cut at their pose, as
        ``cls(shape=..., conform_to=...)``; the returned parameters place it as they place the skin.
        """
        source = cast(Source, catalog.model_spec(cls._name).source)
        names = ("shape", source.pose, source.root_rotation, "global_rotation", "global_translation")
        parameters = {name: arguments.pop(name) for name in names if name in arguments}
        shape, model = parameters.get("shape"), cast(Any, cls)  # a family's constructor: Hand's or SkinnedHumanoid's
        converted = model(**arguments, shape=shape).from_body_models(**parameters)
        return model(**arguments, shape=shape, conform_to=converted[cls._POSE]), converted

    @property
    def shape(self) -> Float[Array, "S"]:
        """Body Models shape coefficients of this model, fixed at construction."""
        return self._conversion.shape

    @property
    def skin_faces(self) -> Int[Array, "F 3"]:
        """Faces of the skin posed by :meth:`forward_vertices`."""
        return self._skin.faces

    def forward_vertices(
        self, *, vertex_indices: Sequence[int] | None = None, **parameters: Float[Array, "..."]
    ) -> Float[Array, "*batch V 3"]:
        """The exact skin of Body Models posed by this model's parameters (those of :meth:`forward_skeleton`).

        Differentiable in Torch and JAX, as Body Models' ``forward_vertices``; ``vertex_indices`` selects vertices.
        """
        return self._skin.forward_vertices(**self.to_body_models(**parameters), vertex_indices=vertex_indices)

    def from_body_models(self, **parameters: Float[Array, "..."]) -> dict[str, Float[Array, "..."]]:
        """Convert Body Models parameters (axis-angle, as ``create_model`` defaults) to this model's.

        The hinge angles reproduce each joint's rotation exactly; the root's rotation and translation
        absorb Body Models' global and root rotations, since this model's root joint sits at the origin.
        A given ``shape`` must be this model's; it is read, so it cannot be traced by JAX.
        """
        xp = self._runtime.xp
        shape = parameters.pop("shape", None)
        own = np.asarray(self._config.shape or 0.0)
        if shape is not None and not np.allclose(self._runtime.to_numpy(shape), own, atol=1e-6):
            raise ValueError(f"{self.name} has shape {own.tolist()}; create a model of the parameters' shape")
        pose = parameters.pop(self._source.pose)
        batch = np.broadcast_shapes(pose.shape[:-2], *(value.shape[:-1] for value in parameters.values()))
        pose = xp.broadcast_to(pose, (*batch, *pose.shape[-2:]))
        zeros = zeros_as(pose, shape=(*batch, 3), xp=xp)
        global_rotation = parameters.pop("global_rotation", zeros)
        root_rotation = parameters.pop(self._source.root_rotation, zeros)
        translation = parameters.pop("global_translation", zeros)
        if parameters:
            raise TypeError(f"{self.name} takes no Body Models parameters {sorted(parameters)}")

        local = SO3.convert(pose[..., self._conversion.pose_rows, :], src="axis_angle", dst="rotmat", xp=xp)
        frames = xp.asarray(self._conversion.hinge_frames, dtype=pose.dtype)
        m = frames.mT @ local @ frames  # intrinsic rotations about the frame's x, y and z axes
        signs = xp.asarray(self._conversion.hinge_signs, dtype=pose.dtype)
        cosine = xp.sqrt(m[..., 0, 0] ** 2 + m[..., 0, 1] ** 2)
        singular = cosine < 1e-6  # the hinge frames are stored in float32
        first = xp.arctan2(
            xp.where(singular, m[..., 2, 1], -m[..., 1, 2]),
            xp.where(singular, m[..., 1, 1], m[..., 2, 2]),
        )
        second = xp.arctan2(m[..., 0, 2], cosine)
        # Recover the third angle from Rx(-first) @ m, stable even close to gimbal lock.
        c, s = xp.cos(first), xp.sin(first)
        third = xp.arctan2(c * m[..., 1, 0] + s * m[..., 2, 0], c * m[..., 1, 1] + s * m[..., 2, 1]) * signs
        angles = xp.stack([first, second, third], axis=-1)
        rotation = SO3.convert(global_rotation, src="axis_angle", dst="rotmat", xp=xp)
        root = SO3.convert(root_rotation, src="axis_angle", dst="rotmat", xp=xp)
        return {
            self._POSE: xp.reshape(angles, (*batch, -1)),
            "global_rotation": SO3.convert(rotation @ root, src="rotmat", dst="axis_angle", xp=xp),
            "global_translation": self._root(rotation) + translation,
        }

    def to_body_models(self, **parameters: Float[Array, "..."]) -> dict[str, Float[Array, "..."]]:
        """Convert this model's parameters to Body Models' (axis-angle), the root rotation global."""
        xp = self._runtime.xp
        pose = parameters.pop(self._POSE)
        rotation_given, translation_given = (
            parameters.pop("global_rotation", None),
            parameters.pop("global_translation", None),
        )
        global_rotation, translation = self._placed(pose, rotation_given, translation_given)
        if parameters:
            raise TypeError(f"{self.name} takes no parameters {sorted(parameters)}")
        batch = np.broadcast_shapes(pose.shape[:-1], global_rotation.shape[:-1], translation.shape[:-1])
        pose = xp.broadcast_to(pose, (*batch, pose.shape[-1]))
        angles = rigid_ops.hinge_angles(self._assets, pose, xp=xp)
        axes = xp.asarray(self._assets.hinge_axes, dtype=pose.dtype)
        hinges = SO3.convert(angles[..., None], src="hinge", dst="rotmat", src_kwargs={"axes": axes}, xp=xp)
        hinges = xp.reshape(hinges, (*batch, -1, 3, 3, 3))
        local = hinges[..., 0, :, :] @ hinges[..., 1, :, :] @ hinges[..., 2, :, :]
        order = np.argsort(self._conversion.pose_rows)  # Body Models' joint order
        rotation = SO3.convert(global_rotation, src="axis_angle", dst="rotmat", xp=xp)
        shape = xp.asarray(self._conversion.shape, dtype=pose.dtype)
        return {
            self._source.pose: SO3.convert(local[..., order, :, :], src="rotmat", dst="axis_angle", xp=xp),
            self._source.root_rotation: zeros_as(pose, shape=(*batch, 3), xp=xp),
            "global_rotation": xp.broadcast_to(global_rotation, (*batch, 3)),
            "global_translation": xp.broadcast_to(translation - self._root(rotation), (*batch, 3)),
            "shape": xp.broadcast_to(shape, (*batch, shape.shape[-1])),
        }

    def _root(self, rotation: Float[Array, "*batch 3 3"]) -> Float[Array, "*batch 3"]:
        """The root joint's rest position in Body Models' frame, rotated."""
        root = self._runtime.xp.asarray(self._conversion.root, dtype=rotation.dtype)
        return self._runtime.xp.squeeze(rotation @ root[..., None], axis=-1)


class SkinnedHand(Skinned, Hand):
    """A skinned hand of a fixed ``shape`` and geometry, posed by ``hand_pose``."""


class SkinnedHumanoid(Skinned, Humanoid):
    """A skinned humanoid of a fixed ``shape`` and geometry, posed by ``body_pose``."""

    def __init__(
        self,
        *,
        gender: Gender = "neutral",
        shape: Sequence[float] | None = None,
        conform_to: Sequence[float] | None = None,
        model_path: Path | str | None = None,
    ) -> None:
        """Load one gender, its geometry fixed by ``shape`` and ``conform_to`` as a skinned hand's."""
        self._setup(gender, model_path, shape, conform_to)

    @property
    def gender(self) -> Gender:
        return cast(Gender, self._config.variant)
