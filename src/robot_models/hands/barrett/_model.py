"""BarrettHand, from its canonical upstream description."""

from types import MappingProxyType

from robot_models.hands._base import Hand


class BarrettHand(Hand):
    """Canonical rigid articulated BarrettHand."""

    _NAME = "barrett"
    supported_sides = ("right",)
    _HAND_JOINTS = MappingProxyType({"wrist": "bh_base_link"})


__all__ = ["BarrettHand"]
