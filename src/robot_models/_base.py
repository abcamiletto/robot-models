from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Any, ClassVar, Literal

from jaxtyping import Float, Int
from nanomanifold import SO3
from trimesh import Trimesh

from robot_models import _pose_layout as pose_layout
from robot_models import _state as state
from robot_models._common import eye_as, zeros_as
from robot_models._common import rigid as rigid_ops
from robot_models._constants import Joint
from robot_models._rotations import RotationType, rotation_ndim, rotation_shape
from robot_models._runtime import ArrayRuntime

Array = Any
ParameterRole = Literal["pose", "transform"]
MUJOCO_TO_MODEL = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


@dataclass(frozen=True)
class ParameterSpec:
    """Shape, role, and numeric default of one model parameter."""

    shape: tuple[int, ...]
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
        leading_shape = () if count is None else (count,)
        return cls(
            shape=(*leading_shape, *rotation_shape(rotation_type)),
            role=role,
            rotation_type=rotation_type,
        )


class RigidBodyModel(ABC):
    """Base class for rigid articulated models."""

    _COMMON_JOINTS: ClassVar[Mapping[Joint, str]] = {}
    _POSE_LAYOUT: ClassVar[pose_layout.PoseLayout | None] = None
    _state_fields: ClassVar[tuple[str, ...]] = ("_weights",)
    _config: Any
    _runtime: ArrayRuntime
    has_face: ClassVar[bool] = False
    has_hands: ClassVar[bool] = False

    @property
    def runtime(self) -> ArrayRuntime:
        """Array runtime used by this model."""
        return self._runtime

    def _attach_runtime(self, runtime: ArrayRuntime) -> None:
        self._runtime = runtime
        if runtime.name == "jax":
            _register_jax_model(type(self))

    def __setstate__(self, values: dict[str, Any]) -> None:
        self.__dict__.update(values)
        if self.runtime.name != "jax":
            return
        _register_jax_model(type(self))
        state.register_jax_state(tuple(getattr(self, name) for name in self._state_fields))

    @property
    def num_joints(self) -> int:
        """Number of joints in the skeleton."""
        return len(self.parents)

    @property
    def common_joints(self) -> Mapping[Joint, str]:
        """Common anatomical joints mapped to this model's native joint names."""
        return self._COMMON_JOINTS

    @property
    def pose_joint_indices(self) -> Mapping[str, tuple[int, ...]]:
        """Canonical joints whose local transforms are driven by each pose parameter."""
        layout = self._pose_layout
        if layout is None:
            return {}
        pose_parameters = {name for name, spec in self.parameter_spec.items() if spec.role == "pose"}
        joint_indices = layout.joint_indices
        if set(joint_indices) != pose_parameters:
            raise ValueError("Pose layout does not match pose parameters")
        return joint_indices

    @property
    def _pose_layout(self) -> pose_layout.PoseLayout | None:
        return self._POSE_LAYOUT

    def joint_index(self, joint: Joint) -> int:
        """Resolve a common joint to this model's native joint index."""
        if not isinstance(joint, Joint):
            raise TypeError("joint_index() expects a robot_models.Joint; use joint_names.index(...) for native names.")
        try:
            native_name = self.common_joints[joint]
        except KeyError as exc:
            raise KeyError(f"{self.__class__.__name__} has no common joint {joint.value!r}") from exc
        return self.joint_names.index(native_name)

    @property
    @abstractmethod
    def parameter_spec(self) -> Mapping[str, ParameterSpec]:
        """Machine-readable parameters accepted by this model."""

    @property
    @abstractmethod
    def _parameter_reference(self) -> Float[Array, "..."]:
        """Array whose backend, device, and dtype parameter defaults follow."""

    @abstractmethod
    def forward_skeleton(self, *args, **kwargs) -> Float[Array, "*batch J 4 4"]:
        """
        Compute skeleton joint transforms.

        Signatures vary by model. Outputs use the model's native coordinate
        system and meters.

        Returns:
            World-space transforms with shape ``[*batch, J, 4, 4]`` in meters.
        """

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
        reference = self._parameter_reference
        if spec.rotation_type is not None:
            encoded_dims = rotation_ndim(spec.rotation_type)
            rotation_batch = spec.shape[:-encoded_dims]
            like = runtime.zeros(batch_dims, like=reference, dtype=dtype)
            return SO3.identity_as(
                like,
                batch_dims=(*batch_dims, *rotation_batch),
                rotation_type=spec.rotation_type,
                xp=runtime.xp,
            )

        value = runtime.zeros((*batch_dims, *spec.shape), like=reference, dtype=dtype)
        return value if spec.default == 0.0 else value + spec.default

    _weights: Any

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self._weights.faces

    @property
    def joint_names(self) -> list[str]:
        return list(self._weights.joint_names)

    @property
    def parents(self) -> list[int]:
        return list(self._weights.parents)

    @property
    def actuated_joint_names(self) -> list[str]:
        return list(self._weights.actuated_joint_names)

    @property
    def actuated_joint_limits(self) -> Float[Array, "Q 2"]:
        return self._weights.actuated_joint_limits

    @property
    def link_names(self) -> list[str]:
        return list(self._weights.link_names)

    @property
    def link_joint_indices(self) -> list[int]:
        return list(self._weights.link_joint_indices)

    @cached_property
    def link_meshes(self) -> Sequence[Trimesh]:
        """Link-local meshes aligned with :attr:`link_names` and ``forward_links()``."""
        return rigid_ops.link_meshes(
            self._weights.vertices,
            self._weights.faces,
            self._weights.link_vertex_starts,
            self._weights.link_vertex_counts,
            self._weights.link_face_starts,
            self._weights.link_face_counts,
            to_numpy=self._runtime.to_numpy,
        )

    @property
    def num_vertices(self) -> int:
        return self._weights.vertices.shape[0]

    @property
    def _parameter_reference(self) -> Float[Array, "V 3"]:
        return self._weights.vertices

    @property
    def num_dofs(self) -> int:
        """Number of scalar pose degrees of freedom."""
        return len(self.actuated_joint_names)

    @property
    def _pose_layout(self) -> pose_layout.PoseLayout:
        (pose_parameter,) = (name for name, spec in self.parameter_spec.items() if spec.role == "pose")
        return pose_layout.PoseLayout(((pose_parameter, self.num_dofs),), self._pose_control_joints)

    @property
    def _pose_control_joints(self) -> tuple[tuple[int, ...], ...]:
        joint_indices = {name: index for index, name in enumerate(self.joint_names)}
        return tuple((joint_indices[name],) for name in self.actuated_joint_names)

    @property
    def actuated_joint_slices(self) -> Mapping[str, slice]:
        """Consecutive scalar coordinate slices keyed by actuated joint name."""
        slices = {}
        seen = set()
        start = 0
        names = self.actuated_joint_names
        while start < len(names):
            name = names[start]
            if name in seen:
                raise ValueError(f"Actuated joint name {name!r} is repeated in non-consecutive groups.")
            seen.add(name)
            stop = start + 1
            while stop < len(names) and names[stop] == name:
                stop += 1
            slices[name] = slice(start, stop)
            start = stop
        return slices

    def unpack_pose(self, pose: Float[Array, "*batch Q"]) -> dict[str, Float[Array, "*batch dof"]]:
        """Unpack a flattened pose ``[..., Q]`` into ``name -> [..., dof]`` arrays."""
        if pose.shape[-1] != self.num_dofs:
            raise ValueError(f"pose must have shape [..., {self.num_dofs}], got {tuple(pose.shape)}")
        return {name: pose[..., joint_slice] for name, joint_slice in self.actuated_joint_slices.items()}

    def pack_pose(self, pose_by_joint: Mapping[str, Float[Array, "*batch dof"]]) -> Float[Array, "*batch Q"]:
        """Pack ``name -> [..., dof]`` arrays into a flattened pose ``[..., Q]``."""
        pieces = []
        expected_names = set(self.actuated_joint_slices)
        extra_names = set(pose_by_joint) - expected_names
        if extra_names:
            raise KeyError(f"Unknown actuated joint names: {sorted(extra_names)}")
        for name, joint_slice in self.actuated_joint_slices.items():
            if name not in pose_by_joint:
                raise KeyError(f"Missing actuated joint name: {name!r}")
            value = pose_by_joint[name]
            dof = joint_slice.stop - joint_slice.start
            if value.shape[-1] != dof:
                raise ValueError(f"{name!r} must have shape [..., {dof}], got {tuple(value.shape)}")
            pieces.append(value)
        return self._runtime.xp.concat(pieces, axis=-1)

    def to_qpos(
        self,
        body_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        clamp_to_limits: bool = False,
    ) -> Float[Array, "*batch qpos"]:
        """Build full MuJoCo ``qpos`` as ``[root_xyz, root_wxyz, body_pose]``.

        ``body_pose`` is the model's flattened scalar coordinate vector ``[..., Q]``.
        ``global_rotation`` is an axis-angle vector ``[..., 3]``.
        The root prefix is converted from the model coordinate frame to MuJoCo's
        coordinate frame.
        """
        if body_pose.shape[-1] != self.num_dofs:
            raise ValueError(f"body_pose must have shape [..., {self.num_dofs}], got {tuple(body_pose.shape)}")

        xp = self._runtime.xp
        batch_shape = tuple(body_pose.shape[:-1])
        if global_translation is None:
            global_translation = zeros_as(body_pose, shape=(*batch_shape, 3), xp=xp)
        if global_rotation is None:
            root_ref = zeros_as(body_pose, shape=(*batch_shape, 3), xp=xp)
            root_rot = eye_as(root_ref, batch_dims=batch_shape, xp=xp)
        else:
            root_rot = SO3.convert(global_rotation, src="axis_angle", dst="rotmat", xp=xp)

        coord = xp.asarray(self._mujoco_to_model(), dtype=body_pose.dtype)
        model_to_mujoco = coord.mT
        root_t = xp.squeeze(model_to_mujoco @ global_translation[..., None], axis=-1)
        root_rot_mujoco = model_to_mujoco @ root_rot @ coord
        root_quat = SO3.conversions.from_rotmat_to_quat(root_rot_mujoco, convention="wxyz", xp=xp)

        if clamp_to_limits:
            limits = xp.asarray(self.actuated_joint_limits, dtype=body_pose.dtype)
            body_pose = xp.clip(body_pose, limits[:, 0], limits[:, 1])
        return xp.concat([root_t, root_quat, body_pose], axis=-1)

    def _mujoco_to_model(self):
        return MUJOCO_TO_MODEL

    @property
    @abstractmethod
    def actuated_joint_types(self) -> list[str]:
        """Actuated pose coordinate types in ``actuated_joint_names`` order."""

    @abstractmethod
    def forward_links(self, *args, **kwargs) -> Float[Array, "*batch L 4 4"]:
        """Compute world-space 4x4 link transforms as the array/autograd primitive."""

    @abstractmethod
    def forward_meshes(self, *args, **kwargs) -> Sequence[Trimesh]:
        """Build one renderer-facing mesh per batch element from link transforms."""

    def _link_transforms(
        self,
        skeleton: Float[Array, "*batch J 4 4"],
    ) -> Float[Array, "*batch L 4 4"]:
        return rigid_ops.forward_link_transforms(
            skeleton,
            self._weights.link_joint_indices,
            self._weights.link_geom_positions,
            self._weights.link_geom_rotations,
            xp=self._runtime.xp,
        )

    def _meshes_from_links(self, links: Float[Array, "*batch L 4 4"]) -> list[Trimesh]:
        return rigid_ops.forward_meshes_from_links(
            links,
            self._weights.vertices,
            self._weights.faces,
            self._weights.link_vertex_starts,
            self._weights.link_vertex_counts,
            self._weights.link_face_starts,
            self._weights.link_face_counts,
            to_numpy=self._runtime.to_numpy,
            xp=self._runtime.xp,
        )


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
