"""Shared behavior for canonical humanoids."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from jaxtyping import Float

from robot_models._base import RigidBodyModel
from robot_models._constants import Joint
from robot_models._runtime import ArrayRuntime

Array = Any


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
