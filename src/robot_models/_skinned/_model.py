"""Rigid robots of Body Models skins, of a shape fixed at construction."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from robot_models import _assets as assets
from robot_models import _catalog as catalog
from robot_models._base import ModelConfig, RigidBodyModel
from robot_models._common import eye_as, zeros_as
from robot_models._common import rigid as rigid_ops
from robot_models._runtime import ArrayRuntime
from robot_models._skinned._geometry import geometry
from robot_models._skinned._sources import Source, source_of
from robot_models._spec import Side
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
    hinge angles, flat by default), cut per joint in its joint frames; its collision pieces follow
    them. Posed at ``conform_to``, the visual links are the skin. ``forward_vertices`` poses the
    exact skin of Body Models at any pose, whose faces are :attr:`skin_faces`.
    """

    _state_fields = ("_assets", "_conversion", "_skin")
    _conversion: Conversion
    _skin: Any

    @property
    def _source(self) -> Source:
        return source_of(self.name)

    def _load(self, xml_name: str, model_path: Path | str | None, runtime: ArrayRuntime) -> None:
        xml_path = assets.model_dir(self.name, model_path) / xml_name
        if not xml_path.is_file():
            raise FileNotFoundError(
                f"{self.name} {self._source.variant_key} {xml_path.stem!r} is not built in {xml_path.parent}: "
                f"configure its Body Models file with `body-models set {self._source.body_model}-{xml_path.stem} PATH`."
            )
        built = geometry(self.name, xml_path, self._config.shape, self._config.conform_to)
        self._set_assets(built.assets, runtime)
        frames = np.swapaxes(built.assets.hinge_axes.reshape(-1, 3, 3), 1, 2).copy()
        signs = np.sign(np.linalg.det(frames)).astype(np.float32)
        frames[..., 2] *= signs[:, None]
        conversion = Conversion(built.pose_rows, frames, signs, built.root, built.shape)
        self._conversion = runtime._materialize(conversion)
        self._skin = self._source.create(xml_path.stem, runtime.name)

    @property
    def shape(self) -> Float[Array, "S"]:
        """Body Models shape coefficients of this model, fixed at construction."""
        return self._conversion.shape

    @property
    def skin_faces(self) -> Int[Array, "F 3"]:
        """Faces of the skin posed by ``forward_vertices()``."""
        return self._skin.faces

    def _forward_vertices(
        self, parameters: dict[str, Any], vertex_indices: Sequence[int] | None
    ) -> Float[Array, "*batch V 3"]:
        """The skin of Body Models posed by this model's parameters (those not ``None``)."""
        given = {key: value for key, value in parameters.items() if value is not None}
        return self._skin.forward_vertices(**self.to_body_models(**given), vertex_indices=vertex_indices)

    def from_body_models(self, **parameters: Float[Array, "..."]) -> dict[str, Float[Array, "..."]]:
        """Convert Body Models parameters (axis-angle, as ``create_model`` defaults) to this model's.

        The hinge angles reproduce each joint's rotation exactly; the root's rotation and
        translation absorb Body Models' global and root rotations, since this model's root
        joint sits at the origin. A given ``shape`` must be this model's (unchecked under JAX tracing).
        """
        xp = self._runtime.xp
        shape = parameters.pop("shape", None)
        if shape is not None and not _traced(shape):
            own = np.asarray(self._config.shape or np.zeros(self.shape.shape[-1]))
            if not np.allclose(self._runtime.to_numpy(shape), own, atol=1e-6):
                raise ValueError(f"{self.name} has shape {own.tolist()}; create a model of the parameters' shape")
        pose_name, root_name = self._POSE, self._source.root_rotation
        pose = parameters[pose_name]
        batch = np.broadcast_shapes(
            pose.shape[:-2], *(value.shape[:-1] for key, value in parameters.items() if key != pose_name)
        )
        pose = xp.broadcast_to(pose, (*batch, *pose.shape[-2:]))
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
        global_rot = self._rotation(parameters.get("global_rotation"), pose, batch)
        root_rot = self._rotation(parameters.get(root_name), pose, batch)
        translation = parameters.get("global_translation")
        root = self._root(global_rot)
        return {
            pose_name: xp.reshape(angles, (*batch, -1)),
            "global_rotation": SO3.convert(global_rot @ root_rot, src="rotmat", dst="axis_angle", xp=xp),
            "global_translation": root if translation is None else root + translation,
        }

    def to_body_models(self, **parameters: Float[Array, "..."]) -> dict[str, Float[Array, "..."]]:
        """Convert this model's parameters to Body Models' (axis-angle), the root rotation global."""
        xp = self._runtime.xp
        pose = parameters[self._POSE]
        batch = np.broadcast_shapes(*(value.shape[:-1] for value in parameters.values()))
        pose = xp.broadcast_to(pose, (*batch, pose.shape[-1]))
        angles = rigid_ops.hinge_angles(self._assets, pose, xp=xp)
        axes = xp.asarray(self._assets.hinge_axes, dtype=pose.dtype)
        hinges = SO3.convert(angles[..., None], src="hinge", dst="rotmat", src_kwargs={"axes": axes}, xp=xp)
        hinges = xp.reshape(hinges, (*batch, -1, 3, 3, 3))
        local = hinges[..., 0, :, :] @ hinges[..., 1, :, :] @ hinges[..., 2, :, :]
        order = np.argsort(self._conversion.pose_rows)  # Body Models' joint order
        body_pose = SO3.convert(local[..., order, :, :], src="rotmat", dst="axis_angle", xp=xp)
        global_rot = self._rotation(parameters.get("global_rotation"), pose, batch)
        translation = parameters.get("global_translation")
        if translation is None:
            translation = zeros_as(pose, shape=(*batch, 3), xp=xp)
        shape = xp.asarray(self._conversion.shape, dtype=pose.dtype)
        return {
            self._POSE: body_pose,
            self._source.root_rotation: zeros_as(pose, shape=(*batch, 3), xp=xp),
            "global_rotation": SO3.convert(global_rot, src="rotmat", dst="axis_angle", xp=xp),
            "global_translation": translation - self._root(global_rot),
            "shape": xp.broadcast_to(shape, (*batch, shape.shape[-1])),
        }

    def _root(self, global_rot: Float[Array, "*batch 3 3"]) -> Float[Array, "*batch 3"]:
        """The root joint's rest position in Body Models' frame, rotated by ``global_rot``."""
        xp = self._runtime.xp
        root = xp.asarray(self._conversion.root, dtype=global_rot.dtype)
        return xp.squeeze(global_rot @ root[..., None], axis=-1)

    def _rotation(self, value: Float[Array, "*batch 3"] | None, like: Array, batch: tuple[int, ...]) -> Array:
        xp = self._runtime.xp
        if value is None:
            return eye_as(zeros_as(like, shape=(*batch, 3), xp=xp), batch_dims=batch, xp=xp)
        return SO3.convert(value, src="axis_angle", dst="rotmat", xp=xp)


def _traced(value: Any) -> bool:
    """Whether a value is traced by JAX, so that its numbers are unknown."""
    jax = sys.modules.get("jax")
    return jax is not None and isinstance(value, jax.core.Tracer)


def _static(values: Any) -> tuple[float, ...] | None:
    """Shape coefficients or a pose as static model identity."""
    return None if values is None else tuple(float(value) for value in np.ravel(values))


class SkinnedHand(Skinned, Hand):
    """A skinned hand of a fixed ``shape``, posed by ``hand_pose``."""

    @property
    def anatomical_limits(self) -> Float[Array, "Q 2"]:
        """Natural ranges of the pose coordinates, after manotorch, in radians.

        Narrower than :attr:`actuated_joint_limits`, which MuJoCo enforces and which reach beyond any
        real hand; a pose outside them is possible but unnatural. Some are empty: a finger's twist.
        """
        anatomy = self._source.anatomy
        return self.runtime.asarray(
            [anatomy[name] for name in self.actuated_joint_names], like=self.actuated_joint_limits
        )

    def __init__(
        self,
        name: str,
        *,
        side: Side = "right",
        shape: Sequence[float] | None = None,
        conform_to: Sequence[float] | None = None,
        model_path: Path | str | None = None,
        runtime: ArrayRuntime,
    ) -> None:
        name = catalog.normalize_name(name)
        sides = catalog.model_spec(name).sides
        if side not in sides:
            raise ValueError(f"{name!r} supports side(s): {sides}")
        self._config = ModelConfig(name, side, shape=_static(shape), conform_to=_static(conform_to))
        self._load(f"{side}.xml", model_path, runtime)

    def forward_vertices(
        self,
        hand_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute the exact skin vertices of Body Models, posed by this model's parameters.

        Differentiable in Torch and JAX, as Body Models' ``forward_vertices``; ``vertex_indices``
        selects vertices.
        """
        parameters = {"hand_pose": hand_pose, "global_rotation": global_rotation}
        parameters["global_translation"] = global_translation
        return self._forward_vertices(parameters, vertex_indices)


class SkinnedHumanoid(Skinned, Humanoid):
    """A skinned humanoid of a fixed ``shape``, posed by ``body_pose``."""

    def __init__(
        self,
        name: str,
        *,
        gender: Gender = "neutral",
        shape: Sequence[float] | None = None,
        conform_to: Sequence[float] | None = None,
        model_path: Path | str | None = None,
        runtime: ArrayRuntime,
    ) -> None:
        name = catalog.normalize_name(name)
        genders = source_of(name).variants
        if gender not in genders:
            raise ValueError(f"{name!r} supports genders: {genders}")
        self._config = ModelConfig(name, variant=gender, shape=_static(shape), conform_to=_static(conform_to))
        self._load(f"{gender}.xml", model_path, runtime)

    @property
    def gender(self) -> Gender:
        return self._config.variant  # type: ignore[return-value]

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute the exact skin vertices of Body Models, posed by this model's parameters.

        Differentiable in Torch and JAX, as Body Models' ``forward_vertices``; ``vertex_indices``
        selects vertices.
        """
        parameters = {"body_pose": body_pose, "global_rotation": global_rotation}
        parameters["global_translation"] = global_translation
        return self._forward_vertices(parameters, vertex_indices)
