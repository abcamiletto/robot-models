"""Fourier GR1 model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaxtyping import Float
from nanomanifold import SO3

from robot_models._base import ParameterSpec, RigidBodyModel
from robot_models._common import coordinates
from robot_models._runtime import ArrayRuntime
from robot_models.gr1 import _core as core
from robot_models.gr1._constants import GR1_BODY_PRESETS, GR1_JOINTS
from robot_models.gr1._io import GR1Assets, load_model_data

Array = Any


@dataclass(frozen=True)
class GR1Config:
    convention: core.Convention


class GR1(RigidBodyModel):
    """Rigid articulated Fourier GR1 model."""

    _COMMON_JOINTS = GR1_JOINTS
    _SIDE_PREFIXES = ("left_", "right_")
    _assets: GR1Assets

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        convention: core.Convention = "soma",
        runtime: ArrayRuntime,
    ) -> None:
        if convention not in ("soma", "mujoco"):
            raise ValueError(f"Invalid GR1 convention: {convention!r}")
        self._attach_runtime(runtime)
        self._config = GR1Config(convention)
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
        """Return the GR1 T-pose."""
        return self._preset_pose("t_pose", batch_dims, dtype)

    def get_apose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the GR1 A-pose."""
        return self._preset_pose("a_pose", batch_dims, dtype)

    def _preset_pose(
        self,
        name: str,
        batch_dims: tuple[int, ...],
        dtype: Any | None,
    ) -> dict[str, Float[Array, "..."]]:
        params = self.get_rest_pose(batch_dims=batch_dims, dtype=dtype)
        runtime = self._runtime
        axis_angle = runtime.asarray(GR1_BODY_PRESETS[name], like=params["body_pose"])
        axis_angle = runtime.xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(
            axis_angle,
            src="axis_angle",
            dst="hinge",
            dst_kwargs={"axes": self._assets.actuated_joint_axes},
            xp=runtime.xp,
        )[..., 0]
        return params


__all__ = ["GR1", "GR1Config"]
