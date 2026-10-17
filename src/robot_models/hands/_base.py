"""Shared behavior for canonical hands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

from jaxtyping import Float

from robot_models._base import RigidBodyModel
from robot_models._constants import Joint
from robot_models._runtime import ArrayRuntime

Array = Any
Side = Literal["left", "right"]


class Hand(RigidBodyModel):
    """Single robot hand posed by ``hand_pose``."""

    _POSE = "hand_pose"
    has_hands = True
    supported_sides: ClassVar[tuple[Side, ...]] = ("right", "left")
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
        if side not in self.supported_sides:
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
