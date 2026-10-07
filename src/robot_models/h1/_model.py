"""Unitree H1 model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaxtyping import Float

from robot_models._base import ParameterSpec, RigidBodyModel
from robot_models._common import coordinates
from robot_models._runtime import ArrayRuntime
from robot_models.h1 import _core as core
from robot_models.h1._constants import H1_BODY_PRESETS, H1_JOINTS
from robot_models.h1._io import H1Assets, load_model_data

Array = Any


@dataclass(frozen=True)
class H1Config:
    convention: core.Convention


class H1(RigidBodyModel):
    """Rigid articulated Unitree H1 model."""

    _COMMON_JOINTS = H1_JOINTS
    _SIDE_PREFIXES = ("left_", "right_")
    _assets: H1Assets

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        convention: core.Convention = "soma",
        runtime: ArrayRuntime,
    ) -> None:
        if convention not in ("soma", "mujoco"):
            raise ValueError(f"Invalid H1 convention: {convention!r}")
        self._attach_runtime(runtime)
        self._config = H1Config(convention)
        self._assets = runtime._materialize(load_model_data(model_path, convention=convention))

    @property
    def convention(self) -> core.Convention:
        return self._config.convention

    def _mujoco_to_model(self):
        if self.convention == "soma":
            return coordinates.MUJOCO_Z_UP_TO_Y_UP
        return super()._mujoco_to_model()

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        return {
            "body_pose": ParameterSpec((self.num_dofs,), "pose"),
            "global_rotation": ParameterSpec.rotation("axis_angle", role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    def forward_skeleton(
        self,
        body_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed joint transforms."""
        assets = self._assets
        return core.forward_skeleton(
            local_offsets=assets.local_offsets,
            rest_local_rotations=assets.rest_local_rotations,
            actuated_joint_indices=assets.actuated_joint_indices,
            actuated_joint_axes=assets.actuated_joint_axes,
            parents=assets.parents,
            body_pose=body_pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
            xp=self._runtime.xp,
        )

    def get_tpose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the H1 T-pose."""
        return self._preset_pose("t_pose", batch_dims, dtype)

    def get_apose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the H1 A-pose."""
        return self._preset_pose("a_pose", batch_dims, dtype)

    def _preset_pose(
        self,
        name: str,
        batch_dims: tuple[int, ...],
        dtype: Any | None,
    ) -> dict[str, Float[Array, "..."]]:
        params = self.get_rest_pose(batch_dims=batch_dims, dtype=dtype)
        body_pose = self._runtime.asarray(H1_BODY_PRESETS[name], like=params["body_pose"])
        params["body_pose"] = self._runtime.xp.broadcast_to(body_pose, (*batch_dims, self.num_dofs))
        return params


__all__ = ["H1", "H1Config"]
