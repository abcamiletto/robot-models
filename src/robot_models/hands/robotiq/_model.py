"""Robotiq3Finger, from its canonical upstream description."""

from types import MappingProxyType

from robot_models._base import Hand


class Robotiq3Finger(Hand):
    """Canonical rigid articulated Robotiq3Finger."""

    _NAME = "robotiq-3finger"
    supported_sides = ("right",)
    _HAND_JOINTS = MappingProxyType({"wrist": "palm"})


__all__ = ["Robotiq3Finger"]
