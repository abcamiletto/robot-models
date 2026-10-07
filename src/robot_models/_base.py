"""Rigid articulated models loaded from MuJoCo MJCF files."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, cached_property, partial
from importlib.resources import files
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3
from trimesh import Trimesh

from robot_models import _assets as assets
from robot_models import _state as state
from robot_models._common import eye_as, mjcf, zeros_as
from robot_models._common import rigid as rigid_ops
from robot_models._constants import Joint
from robot_models._rotations import RotationType, rotation_dims, rotation_ndim
from robot_models._runtime import ArrayRuntime

Array = Any
ParameterRole = Literal["pose", "transform"]
Side = Literal["left", "right"]


@dataclass(frozen=True)
class ParameterSpec:
    """Array dims, role, and numeric default of one model parameter."""

    dims: tuple[int, ...]
    role: ParameterRole
    default: float = field(default=0.0, kw_only=True)
    rotation_type: RotationType | None = field(default=None, kw_only=True)

    @classmethod
    def rotation(
        cls,
        rotation_type: RotationType,
        *,
        count: int | None = None,
        role: ParameterRole = "pose",
    ) -> ParameterSpec:
        """Describe one rotation or a vector of rotations."""
        leading_dims = () if count is None else (count,)
        return cls(
            dims=(*leading_dims, *rotation_dims(rotation_type)),
            role=role,
            rotation_type=rotation_type,
        )


class RigidBodyModel(ABC):
    """Rigid articulated model loaded from one MJCF file.

    Skeleton joints are the MJCF bodies and pose coordinates are its independent
    hinges, both in MuJoCo order. Outputs are Y-up and in meters.
    """

    # Asset folder on Hugging Face and package holding ``assets/poses.npz``.
    _NAME: ClassVar[str]
    _POSE: ClassVar[str]
    # Left/right name fragments that pair mirrored joints.
    _MIRROR: ClassVar[tuple[tuple[str, str], ...]] = ()
    _state_fields: ClassVar[tuple[str, ...]] = ("_assets",)
    has_hands: ClassVar[bool] = False
    _config: Any
    _runtime: ArrayRuntime
    _assets: rigid_ops.RigidAssets

    def _load(self, xml_name: str, model_path: Path | str | None, runtime: ArrayRuntime) -> None:
        self._runtime = runtime
        if runtime.name == "jax":
            _register_jax_model(type(self))
        xml_path = assets.model_dir(self._NAME, model_path) / xml_name
        self._assets = runtime._materialize(mjcf.load(xml_path))

    @property
    def runtime(self) -> ArrayRuntime:
        """Array runtime used by this model."""
        return self._runtime

    def __setstate__(self, values: dict[str, Any]) -> None:
        self.__dict__.update(values)
        if self.runtime.name != "jax":
            return
        _register_jax_model(type(self))
        state.register_jax_state(tuple(getattr(self, name) for name in self._state_fields))

    @property
    @abstractmethod
    def common_joints(self) -> Mapping[Joint, str]:
        """Common anatomical joints mapped to this model's native joint names."""

    @abstractmethod
    def forward_skeleton(self, *args, **kwargs) -> Float[Array, "*batch J 4 4"]:
        """Compute world-space joint transforms with shape ``[*batch, J, 4, 4]``."""

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        """Machine-readable parameters accepted by this model."""
        return {
            self._POSE: ParameterSpec((self.num_dofs,), "pose"),
            "global_rotation": ParameterSpec.rotation("axis_angle", role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    @property
    def num_joints(self) -> int:
        """Number of joints in the skeleton."""
        return len(self.parents)

    @property
    def num_dofs(self) -> int:
        """Number of scalar pose degrees of freedom."""
        return len(self.actuated_joint_names)

    @property
    def symmetric_joints(self) -> tuple[tuple[int, int], ...]:
        """
        Left/right joint pairs as ``(left_index, right_index)``, in joint order.

        Indices address the ``J`` axis of :meth:`forward_skeleton` outputs and
        cover the whole native skeleton. Unpaired joints lie on the midline.

        Raises:
            ValueError: If a sided joint name has no counterpart.
        """
        names = self.joint_names
        pairs = []
        for index, name in enumerate(names):
            mirrored = name
            for left, right in self._MIRROR:
                mirrored = mirrored.replace(left, right)
            if mirrored != name:
                if mirrored not in names:
                    raise ValueError(f"{type(self).__name__} joint {name!r} has no counterpart {mirrored!r}")
                pairs.append((index, names.index(mirrored)))
        return tuple(pairs)

    @property
    def pose_joint_indices(self) -> Mapping[str, tuple[int, ...]]:
        """Skeleton joints whose local transforms are driven by each pose parameter."""
        driven = sorted(zip(self._assets.hinge_drivers, self._assets.hinge_joint_indices, strict=True))
        return {self._POSE: tuple(dict.fromkeys(joint for _, joint in driven))}

    def joint_index(self, joint: Joint) -> int:
        """Resolve a common joint to this model's native joint index."""
        if not isinstance(joint, Joint):
            raise TypeError("joint_index() expects a robot_models.Joint; use joint_names.index(...) for native names.")
        try:
            native_name = self.common_joints[joint]
        except KeyError as exc:
            raise KeyError(f"{type(self).__name__} has no common joint {joint.value!r}") from exc
        return self.joint_names.index(native_name)

    def get_rest_pose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
    ) -> dict[str, Float[Array, "..."]]:
        """
        Construct canonical parameter defaults from :attr:`parameter_spec`.

        Args:
            batch_dims: Leading batch dimensions.
            dtype: Optional floating-point dtype.

        Returns:
            Complete model parameters at rest.
        """
        return {name: self._parameter_default(spec, batch_dims, dtype) for name, spec in self.parameter_spec.items()}

    def _parameter_default(
        self,
        spec: ParameterSpec,
        batch_dims: tuple[int, ...],
        dtype: Any | None,
    ) -> Float[Array, "..."]:
        runtime = self.runtime
        reference = self._assets.vertices
        if spec.rotation_type is not None:
            encoded_dims = rotation_ndim(spec.rotation_type)
            rotation_batch = spec.dims[:-encoded_dims]
            like = runtime.zeros(batch_dims, like=reference, dtype=dtype)
            return SO3.identity_as(
                like,
                batch_dims=(*batch_dims, *rotation_batch),
                rotation_type=spec.rotation_type,
                xp=runtime.xp,
            )

        value = runtime.zeros((*batch_dims, *spec.dims), like=reference, dtype=dtype)
        return value if spec.default == 0.0 else value + spec.default

    def _preset(self, name: str, batch_dims: tuple[int, ...], dtype: Any | None) -> dict[str, Float[Array, "..."]]:
        params = RigidBodyModel.get_rest_pose(self, batch_dims=batch_dims, dtype=dtype)
        pose = self._runtime.asarray(_presets(self._NAME)[name], like=params[self._POSE])
        params[self._POSE] = self._runtime.xp.broadcast_to(pose, (*batch_dims, self.num_dofs))
        return params

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self._assets.faces

    @property
    def joint_names(self) -> list[str]:
        return list(self._assets.joint_names)

    @property
    def parents(self) -> list[int]:
        return list(self._assets.parents)

    @property
    def actuated_joint_names(self) -> list[str]:
        return list(self._assets.actuated_joint_names)

    @property
    def actuated_joint_limits(self) -> Float[Array, "Q 2"]:
        return self._assets.actuated_joint_limits

    @property
    def actuated_joint_types(self) -> list[str]:
        """Pose coordinate types in ``actuated_joint_names`` order."""
        return ["hinge"] * self.num_dofs

    @property
    def link_names(self) -> list[str]:
        return list(self._assets.link_names)

    @property
    def link_joint_indices(self) -> list[int]:
        return list(self._assets.link_joint_indices)

    @cached_property
    def link_meshes(self) -> Sequence[Trimesh]:
        """Meshes in their joint's frame, aligned with :attr:`link_names` and ``forward_links()``."""
        return rigid_ops.link_meshes(
            self._assets.vertices,
            self._assets.faces,
            self._assets.link_vertex_starts,
            self._assets.link_vertex_counts,
            self._assets.link_face_starts,
            self._assets.link_face_counts,
            to_numpy=self._runtime.to_numpy,
        )

    @property
    def num_vertices(self) -> int:
        return self._assets.vertices.shape[0]

    def unpack_pose(self, pose: Float[Array, "*batch Q"]) -> dict[str, Float[Array, "*batch 1"]]:
        """Unpack a pose ``[..., Q]`` into ``name -> [..., 1]`` arrays."""
        if pose.shape[-1] != self.num_dofs:
            raise ValueError(f"pose must have shape [..., {self.num_dofs}], got {tuple(pose.shape)}")
        return {name: pose[..., index : index + 1] for index, name in enumerate(self.actuated_joint_names)}

    def pack_pose(self, pose_by_joint: Mapping[str, Float[Array, "*batch 1"]]) -> Float[Array, "*batch Q"]:
        """Pack ``name -> [..., 1]`` arrays into a pose ``[..., Q]``."""
        if set(pose_by_joint) != set(self.actuated_joint_names):
            raise KeyError(f"Expected exactly the actuated joints {self.actuated_joint_names}")
        return self._runtime.xp.concat([pose_by_joint[name] for name in self.actuated_joint_names], axis=-1)

    def to_qpos(
        self,
        pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        clamp_to_limits: bool = False,
    ) -> Float[Array, "*batch qpos"]:
        """Build MuJoCo ``qpos`` as ``[root_xyz, root_wxyz, hinge_angles]``.

        The root prefix is the free joint of the MJCF root body, converted to MuJoCo's
        Z-up frame. Hinge angles include coupled hinges, in MJCF order.
        """
        if pose.shape[-1] != self.num_dofs:
            raise ValueError(f"pose must have shape [..., {self.num_dofs}], got {tuple(pose.shape)}")
        xp = self._runtime.xp
        batch_shape = tuple(pose.shape[:-1])
        if global_translation is None:
            global_translation = zeros_as(pose, shape=(*batch_shape, 3), xp=xp)
        if global_rotation is None:
            root_rot = eye_as(zeros_as(pose, shape=(*batch_shape, 3), xp=xp), batch_dims=batch_shape, xp=xp)
        else:
            root_rot = SO3.convert(global_rotation, src="axis_angle", dst="rotmat", xp=xp)
        if clamp_to_limits:
            limits = xp.asarray(self.actuated_joint_limits, dtype=pose.dtype)
            pose = xp.clip(pose, limits[:, 0], limits[:, 1])

        to_model = xp.asarray(mjcf.MUJOCO_TO_MODEL, dtype=pose.dtype)
        root_t = xp.squeeze(to_model.mT @ global_translation[..., None], axis=-1)
        root_quat = SO3.conversions.from_rotmat_to_quat(to_model.mT @ root_rot @ to_model, convention="wxyz", xp=xp)
        angles = rigid_ops.hinge_angles(self._assets, pose, xp=xp)
        return xp.concat([root_t, root_quat, angles], axis=-1)

    def forward_links(self, **parameters: Float[Array, "..."]) -> Float[Array, "*batch L 4 4"]:
        """Compute the world transform of each link mesh's joint frame."""
        return self.forward_skeleton(**parameters)[..., self._assets.link_joint_indices, :, :]

    def forward_meshes(self, **parameters: Float[Array, "..."]) -> list[Trimesh]:
        """Build one renderer-facing mesh per batch element."""
        return rigid_ops.forward_meshes_from_links(
            self.forward_links(**parameters),
            self._assets.vertices,
            self._assets.faces,
            self._assets.link_vertex_starts,
            self._assets.link_vertex_counts,
            self._assets.link_face_starts,
            self._assets.link_face_counts,
            to_numpy=self._runtime.to_numpy,
            xp=self._runtime.xp,
        )

    def _forward_skeleton(
        self,
        pose: Float[Array, "*batch Q"],
        global_rotation: Float[Array, "*batch 3"] | None,
        global_translation: Float[Array, "*batch 3"] | None,
        joint_indices: Sequence[int] | None,
    ) -> Float[Array, "*batch J 4 4"]:
        return rigid_ops.forward_skeleton(
            self._assets,
            pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            xp=self._runtime.xp,
        )


class Humanoid(RigidBodyModel):
    """Floating-base humanoid posed by ``body_pose``."""

    _POSE = "body_pose"
    _MIRROR = (("left", "right"),)
    _COMMON_JOINTS: ClassVar[Mapping[Joint, str]]

    def __init__(self, *, model_path: Path | str | None = None, runtime: ArrayRuntime) -> None:
        self._config = None
        self._load(f"{self._NAME}.xml", model_path, runtime)

    @property
    def common_joints(self) -> Mapping[Joint, str]:
        return self._COMMON_JOINTS

    def forward_skeleton(
        self,
        body_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed joint transforms."""
        return self._forward_skeleton(body_pose, global_rotation, global_translation, joint_indices)

    def get_tpose(
        self, *, batch_dims: tuple[int, ...] = (), dtype: Any | None = None
    ) -> dict[str, Float[Array, "..."]]:
        """Return the T-pose."""
        return self._preset("t_pose", batch_dims, dtype)

    def get_apose(
        self, *, batch_dims: tuple[int, ...] = (), dtype: Any | None = None
    ) -> dict[str, Float[Array, "..."]]:
        """Return the A-pose."""
        return self._preset("a_pose", batch_dims, dtype)


class Hand(RigidBodyModel):
    """Single robot hand posed by ``hand_pose``."""

    _POSE = "hand_pose"
    has_hands = True
    # Native names keyed by ``Joint`` without its side. In names, ``{side}``
    # expands to ``left``/``right`` and ``{s}`` to ``l``/``r``.
    _HAND_JOINTS: ClassVar[Mapping[str, str]]

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        side: Side = "right",
        runtime: ArrayRuntime,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"Invalid side: {side!r}")
        self._config = side
        self._load(f"{side}.xml", model_path, runtime)

    @property
    def side(self) -> Side:
        return self._config

    @property
    def common_joints(self) -> Mapping[Joint, str]:
        side = self.side
        return {Joint(f"{side}_{key}"): name.format(side=side, s=side[0]) for key, name in self._HAND_JOINTS.items()}

    def forward_skeleton(
        self,
        hand_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed joint transforms."""
        return self._forward_skeleton(hand_pose, global_rotation, global_translation, joint_indices)

    def get_rest_pose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the zero pose (``"default"``) or the ``"flat"`` or ``"rest"`` hand preset."""
        if hands == "default":
            return super().get_rest_pose(batch_dims=batch_dims, dtype=dtype)
        return self._preset(f"{self.side}_{hands}", batch_dims, dtype)


@cache
def _presets(name: str) -> dict[str, Float[np.ndarray, "Q"]]:
    with (files(f"robot_models.{name}") / "assets" / "poses.npz").open("rb") as file:
        return dict(np.load(file))


_JAX_MODELS: set[type] = set()


def _register_jax_model(model_type: type) -> None:
    if model_type in _JAX_MODELS:
        return
    import jax

    jax.tree_util.register_pytree_node(
        model_type,
        _flatten_model,
        partial(_unflatten_model, model_type),
    )
    _JAX_MODELS.add(model_type)


def _flatten_model(model: RigidBodyModel) -> tuple[tuple[Any, ...], tuple[Any, ArrayRuntime]]:
    children = tuple(getattr(model, name) for name in model._state_fields)
    return children, (model._config, model._runtime)


def _unflatten_model(
    model_type: type[RigidBodyModel],
    auxiliary: tuple[Any, ArrayRuntime],
    children: tuple[Any, ...],
) -> RigidBodyModel:
    config, runtime = auxiliary
    model = model_type.__new__(model_type)
    model._runtime = runtime
    model._config = config
    for name, value in zip(model_type._state_fields, children, strict=True):
        setattr(model, name, value)
    return model
