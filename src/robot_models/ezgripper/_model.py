"""EZGripper, from its canonical upstream description."""

from types import MappingProxyType

from robot_models._base import Hand


class EZGripper(Hand):
    """Canonical rigid articulated EZGripper."""

    _NAME = "ezgripper"
    supported_sides = ("right",)
    _HAND_JOINTS = MappingProxyType({"wrist": "mount"})


__all__ = ["EZGripper"]
