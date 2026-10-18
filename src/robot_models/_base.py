"""Rigid articulated models loaded from MuJoCo MJCF files."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache, partial
from importlib.resources import files
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
from jaxtyping import Float
from nanomanifold import SO3

from robot_models import _assets as assets
from robot_models import _catalog as catalog
from robot_models import _state as state
from robot_models._common import mjcf, zeros_as
from robot_models._common import rigid as rigid_ops
from robot_models._common.rigid import MeshSet
from robot_models._constants import Joint
from robot_models._runtime import ArrayRuntime

Array = Any


@dataclass(frozen=True)
class ModelConfig:
    """Static identity preserved when a model is serialized or passed through JAX.

    ``variant`` is the MJCF file the model loads (a hand's side, SMPL's gender), ``directory``
    holds it; ``shape`` and ``conform_to`` fix a skinned robot's geometry.
    """

    name: str
    variant: str | None
    directory: Path
    shape: tuple[float, ...] | None = None
    conform_to: tuple[float, ...] | None = None

    @property
    def xml_path(self) -> Path:
        return self.directory / f"{self.variant or self.name}.xml"


class RigidBodyModel(ABC):
    """Rigid articulated model loaded from one MJCF file.

    Its bodies are the MJCF bodies and its pose coordinates, the dofs, the independent hinges, both
    in MuJoCo order. Outputs are Y-up and in meters. Each catalog model's class is built for one
    runtime by the class factory, which sets ``_name`` and ``_runtime``.
    """

    _name: ClassVar[str]
    _runtime: ArrayRuntime
    _POSE: ClassVar[str]
    _state_fields: ClassVar[tuple[str, ...]] = ("_assets",)
    _config: ModelConfig
    _assets: rigid_ops.RigidAssets

    def _setup(self, variant: str | None, model_path: Path | str | None, shape: Any, conform_to: Any) -> None:
        if self._runtime.name == "torch":
            from torch import nn

            nn.Module.__init__(cast(Any, self))  # the class factory's torch classes are modules
        if self._runtime.name == "jax":
            _register_jax_model(type(self))
        variants = catalog.model_spec(self._name).variants
        if variant not in (variants or (None,)):
            raise ValueError(f"{self._name!r} offers {variants}, not {variant!r}")
        directory = assets.model_dir(self._name, model_path)
        self._config = ModelConfig(self._name, variant, directory, _static(shape), _static(conform_to))
        self._load()

    def _load(self) -> None:
        if self._config.shape is not None or self._config.conform_to is not None:
            raise ValueError(f"{self.name} has no Body Models shape or conformed geometry")
        self._assets = self._runtime._materialize(mjcf.load(self._config.xml_path))

    def __setstate__(self, values: dict[str, Any]) -> None:
        self.__dict__.update(values)
        if self._runtime.name == "jax":
            _register_jax_model(type(self))
            for name in self._state_fields:
                state.register_jax_state(getattr(self, name))

    @property
    def runtime(self) -> ArrayRuntime:
        """Array runtime used by this model."""
        return self._runtime

    @property
    def name(self) -> str:
        """Canonical model identifier."""
        return self._config.name

    @property
    def xml_path(self) -> Path:
        """The MJCF file this model loads, for ``mujoco.MjModel.from_xml_path``; :func:`add_hand` mounts a hand's."""
        return self._config.xml_path

    @property
    @abstractmethod
    def common_joints(self) -> Mapping[Joint, str]:
        """Common anatomical joints mapped to this model's body names."""

    def body_index(self, joint: Joint) -> int:
        """The index in :attr:`body_names` of a common anatomical joint's body."""
        try:
            return self.body_names.index(self.common_joints[joint])
        except KeyError as exc:
            raise KeyError(f"{type(self).__name__} has no common joint {joint.value!r}") from exc

    @property
    def body_names(self) -> list[str]:
        """MJCF bodies, in MuJoCo order: the frames of :meth:`forward_skeleton`."""
        return list(self._assets.body_names)

    @property
    def parents(self) -> list[int]:
        """Each body's parent in :attr:`body_names`, -1 for the root."""
        return list(self._assets.parents)

    @property
    def num_bodies(self) -> int:
        return len(self._assets.body_names)

    @property
    def dof_names(self) -> list[str]:
        """The independent hinges, in MuJoCo order: the pose coordinates."""
        return list(self._assets.dof_names)

    @property
    def num_dofs(self) -> int:
        return len(self._assets.dof_names)

    @property
    def dof_limits(self) -> Float[Array, "Q 2"]:
        """What each pose coordinate reaches in MuJoCo, in radians: its enforced range, narrowed where coupled
        hinges stop it; ``inf`` where unlimited."""
        return self._assets.dof_limits

    @property
    def hinge_names(self) -> list[str]:
        """Every MJCF hinge, coupled ones included, in MuJoCo order: the angles of :meth:`to_qpos`."""
        return list(self._assets.hinge_names)

    @property
    def visual(self) -> MeshSet:
        """Visual meshes, each rigid in its body's frame."""
        return MeshSet(self._assets.visual, self._runtime)

    @property
    def collision(self) -> MeshSet:
        """Collision meshes, each rigid in its body's frame."""
        return MeshSet(self._assets.collision, self._runtime)

    @property
    def parameter_spec(self) -> dict[str, tuple[int, ...]]:
        """The dims of each parameter of the forward pass, axis-angle for ``global_rotation``."""
        return {self._POSE: (self.num_dofs,), "global_rotation": (3,), "global_translation": (3,)}

    def get_rest_pose(
        self, *, preset: str | None = None, batch_dims: tuple[int, ...] = (), dtype: Any | None = None
    ) -> dict[str, Float[Array, "..."]]:
        """Parameters at rest: all zeros, or a preset pose (hands: ``flat``, ``rest``, ``open``, ``closed``;
        humanoids: ``t_pose``, ``a_pose``) at the origin."""
        xp, like = self._runtime.xp, self._assets.local_offsets
        params = {
            name: zeros_as(like, shape=(*batch_dims, *dims), dtype=dtype, xp=xp)
            for name, dims in self.parameter_spec.items()
        }
        if preset is not None:
            presets = _presets(self.name)
            key = self._preset_key(preset)
            if key not in presets:
                raise KeyError(f"{self.name} has no preset {preset!r}: {sorted(presets)}")
            pose = self._runtime.asarray(presets[key], like=params[self._POSE])
            params[self._POSE] = self._runtime.xp.broadcast_to(pose, params[self._POSE].shape)
        return params

    def _preset_key(self, preset: str) -> str:
        return preset

    def _placed(
        self,
        pose: Float[Array, "*batch Q"],
        global_rotation: Float[Array, "*batch 3"] | None,
        global_translation: Float[Array, "*batch 3"] | None,
    ) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
        """The root's axis-angle rotation and translation, zeros where not given, for a checked pose."""
        self._check(pose)
        zeros = zeros_as(pose, shape=(*pose.shape[:-1], 3), xp=self._runtime.xp)
        rotation = zeros if global_rotation is None else global_rotation
        translation = zeros if global_translation is None else global_translation
        return rotation, translation

    def _check(self, pose: Float[Array, "*batch Q"]) -> None:
        if pose.shape[-1:] != (self.num_dofs,):
            raise ValueError(f"{self._POSE} must have shape [..., {self.num_dofs}], got {tuple(pose.shape)}")

    def _forward_skeleton(
        self, pose: Float[Array, "*batch Q"], global_rotation: Array, global_translation: Array
    ) -> Float[Array, "*batch B 4 4"]:
        rotation, translation = self._placed(pose, global_rotation, global_translation)
        return rigid_ops.forward_skeleton(self._assets, pose, rotation, translation, xp=self._runtime.xp)

    def _to_qpos(
        self,
        pose: Float[Array, "*batch Q"],
        global_rotation: Array,
        global_translation: Array,
        scene_transform: Float[Array, "*batch 4 4"] | None,
    ) -> Float[Array, "*batch qpos"]:
        xp = self._runtime.xp
        rotation, translation = self._placed(pose, global_rotation, global_translation)
        root = SO3.convert(rotation, src="axis_angle", dst="rotmat", xp=xp)
        if scene_transform is not None:
            root = scene_transform[..., :3, :3] @ root
            translation = xp.squeeze(scene_transform[..., :3, :3] @ translation[..., None], axis=-1)
            translation = translation + scene_transform[..., :3, 3]
        to_model = xp.asarray(mjcf.MUJOCO_TO_MODEL, dtype=root.dtype)
        quaternion = SO3.conversions.from_rotmat_to_quat(root @ to_model, convention="wxyz", xp=xp)
        angles = rigid_ops.hinge_angles(self._assets, pose, xp=xp)
        return xp.concat([translation, quaternion, angles], axis=-1)


def _static(values: Any) -> tuple[float, ...] | None:
    """Shape coefficients or a pose as static model identity."""
    return None if values is None else tuple(float(value) for value in np.ravel(values))


@cache
def _presets(name: str) -> dict[str, Float[np.ndarray, "Q"]]:
    with (files(catalog.package(name)) / "assets" / "poses.npz").open("rb") as file:
        return dict(np.load(file))


_JAX_MODELS: set[type] = set()


def _register_jax_model(model_type: type) -> None:
    if model_type in _JAX_MODELS:
        return
    import jax

    jax.tree_util.register_pytree_node(model_type, _flatten_model, partial(_unflatten_model, model_type))
    _JAX_MODELS.add(model_type)


def _flatten_model(model: RigidBodyModel) -> tuple[tuple[Any, ...], tuple[Any, ArrayRuntime]]:
    children = tuple(getattr(model, name) for name in model._state_fields)
    return children, (model._config, model._runtime)


def _unflatten_model(
    model_type: type[RigidBodyModel], auxiliary: tuple[Any, ArrayRuntime], children: tuple[Any, ...]
) -> RigidBodyModel:
    config, runtime = auxiliary
    model = model_type.__new__(model_type)
    model._runtime = runtime
    model._config = config
    for name, value in zip(model_type._state_fields, children, strict=True):
        setattr(model, name, value)
    return model
