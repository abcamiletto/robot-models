"""Baseline-native hands with source joint names, signs, order, and rest frames."""

from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from robot_models._base import Hand, Side
from robot_models._runtime import ArrayRuntime


class BaselineHand(Hand):
    """Single source hand; no reflection or implicit canonical joint remapping."""

    supported_sides: tuple[Side, ...] = ("right",)

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        side: Side | None = None,
        runtime: ArrayRuntime,
    ) -> None:
        side = self.supported_sides[0] if side is None else side
        if side not in self.supported_sides:
            raise ValueError(f"{self._NAME} only provides source side(s): {self.supported_sides}")
        super().__init__(model_path=model_path, side=side, runtime=runtime)

    def get_rest_pose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Any]:
        """Return source zero, or zero clipped to source limits for flat/rest."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Unknown hand preset: {hands!r}")
        params = super().get_rest_pose(batch_dims=batch_dims, dtype=dtype)
        if hands != "default":
            lo, hi = self.actuated_joint_limits[:, 0], self.actuated_joint_limits[:, 1]
            params["hand_pose"] = self.runtime.xp.clip(params["hand_pose"], lo, hi)
        return params


class BarrettGenDexGrasp(BaselineHand):
    """GenDexGrasp's Barrett AdaGrasp source, including eight independent hinges."""

    _NAME = "barrett-gendexgrasp"
    _HAND_JOINTS = MappingProxyType({"wrist": "bh_base_link"})


class Robotiq3FingerGenDexGrasp(BaselineHand):
    """GenDexGrasp's eleven-coordinate Robotiq three-finger source."""

    _NAME = "robotiq-3finger-gendexgrasp"
    _HAND_JOINTS = MappingProxyType({"wrist": "gripper_palm"})


class EZGripperGenDexGrasp(BaselineHand):
    """GenDexGrasp's four-coordinate EZGripper source; left assembly only."""

    _NAME = "ezgripper-gendexgrasp"
    supported_sides = ("left",)
    _HAND_JOINTS = MappingProxyType({"wrist": "left_ezgripper_palm_link"})
