"""Shared behavior for canonical humanoids."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jaxtyping import Float

from robot_models import _catalog as catalog
from robot_models._base import ModelConfig, RigidBodyModel
from robot_models._constants import Joint
from robot_models._runtime import ArrayRuntime

Array = Any


class Humanoid(RigidBodyModel):
    """Floating-base humanoid posed by ``body_pose``."""

    _POSE = "body_pose"
    _MIRROR = (("left", "right"),)

    def __init__(self, name: str, *, model_path: Path | str | None = None, runtime: ArrayRuntime) -> None:
        name = catalog.normalize_name(name)
        if catalog.model_spec(name).family != "humanoids":
            raise ValueError(f"{name!r} is not a humanoid")
        self._config = ModelConfig(name)
        self._load(f"{name}.xml", model_path, runtime)

    @property
    def common_joints(self) -> Mapping[Joint, str]:
        return {Joint(key): value for key, value in catalog.model_spec(self.name).joints.items()}

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
