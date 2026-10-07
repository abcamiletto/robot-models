"""Wonik Robotics Allegro hand model implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float

from robot_models._base import ParameterSpec, RigidBodyModel
from robot_models._common import coordinates
from robot_models._constants import Joint
from robot_models._runtime import ArrayRuntime
from robot_models.allegro import _core as core
from robot_models.allegro._constants import ALLEGRO_HAND_PRESETS, LEFT_ALLEGRO_JOINTS, RIGHT_ALLEGRO_JOINTS
from robot_models.allegro._io import AllegroAssets, Side, load_model_data

Array = Any


@dataclass(frozen=True)
class AllegroConfig:
    side: Side


class AllegroHand(RigidBodyModel):
    """Rigid articulated Wonik Robotics Allegro hand."""

    _assets: AllegroAssets
    has_hands = True

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        side: Side = "right",
        runtime: ArrayRuntime,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"Invalid side: {side!r}")
        self._attach_runtime(runtime)
        self._config = AllegroConfig(side)
        self._assets = runtime._materialize(load_model_data(model_path, side=side))

    @property
    def side(self) -> Side:
        return self._config.side

    @property
    def common_joints(self) -> Mapping[Joint, str]:
        return LEFT_ALLEGRO_JOINTS if self.side == "left" else RIGHT_ALLEGRO_JOINTS

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        return {
            "hand_pose": ParameterSpec((self.num_dofs,), "pose"),
            "global_rotation": ParameterSpec.rotation("axis_angle", role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    def _mujoco_to_model(self):
        return coordinates.MUJOCO_Z_UP_TO_Y_UP

    def forward_skeleton(
        self,
        hand_pose: Float[Array, "*batch Q"],
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
            hand_pose=hand_pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
            xp=self._runtime.xp,
        )

    def get_rest_pose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the configured default or canonical hand pose."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")
        params = super().get_rest_pose(batch_dims=batch_dims, dtype=dtype)
        if hands != "default":
            hand_pose = self._runtime.asarray(ALLEGRO_HAND_PRESETS[self.side][hands], like=params["hand_pose"])
            params["hand_pose"] = self._runtime.xp.broadcast_to(hand_pose, (*batch_dims, self.num_dofs))
        return params


__all__ = ["AllegroConfig", "AllegroHand"]
