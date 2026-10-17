"""Shared behavior for canonical hands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from jaxtyping import Float

from robot_models import _catalog as catalog
from robot_models._base import ModelConfig, RigidBodyModel
from robot_models._constants import Joint
from robot_models._runtime import ArrayRuntime
from robot_models._spec import Side

Array = Any


class Hand(RigidBodyModel):
    """Single robot hand posed by ``hand_pose``."""

    _POSE = "hand_pose"
    has_hands = True

    def __init__(
        self,
        name: str,
        *,
        model_path: Path | str | None = None,
        side: Side = "right",
        runtime: ArrayRuntime,
    ) -> None:
        name = catalog.normalize_name(name)
        definition = catalog.model_spec(name)
        if definition.family != "hands":
            raise ValueError(f"{name!r} is not a hand")
        if side not in definition.sides:
            raise ValueError(f"{name!r} supports side(s): {definition.sides}")
        self._config = ModelConfig(name, side)
        self._load(f"{side}.xml", model_path, runtime)

    @property
    def side(self) -> Side:
        return cast(Side, self._config.side)

    @property
    def supported_sides(self) -> tuple[Side, ...]:
        """Sides supplied by this hand's canonical description."""
        return catalog.model_spec(self.name).sides

    @property
    def common_joints(self) -> Mapping[Joint, str]:
        side = self.side
        return {
            Joint(f"{side}_{key}"): name.format(side=side, s=side[0])
            for key, name in catalog.model_spec(self.name).joints.items()
        }

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
