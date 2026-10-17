"""Sharpa Wave dexterous hand."""

from types import MappingProxyType

from robot_models._base import Hand


class SharpaHand(Hand):
    """Rigid articulated Sharpa Wave hand with 22 independent hinge coordinates."""

    _NAME = "sharpa"
    _HAND_JOINTS = MappingProxyType(
        {
            "wrist": "{side}_hand_C_MC",
            "thumb_cmc": "{side}_thumb_MC",
            "thumb_mcp": "{side}_thumb_PP",
            "thumb_ip": "{side}_thumb_DP",
            "index_mcp": "{side}_index_PP",
            "index_pip": "{side}_index_MP",
            "index_dip": "{side}_index_DP",
            "middle_mcp": "{side}_middle_PP",
            "middle_pip": "{side}_middle_MP",
            "middle_dip": "{side}_middle_DP",
            "ring_mcp": "{side}_ring_PP",
            "ring_pip": "{side}_ring_MP",
            "ring_dip": "{side}_ring_DP",
            "pinky_mcp": "{side}_pinky_PP",
            "pinky_pip": "{side}_pinky_MP",
            "pinky_dip": "{side}_pinky_DP",
        }
    )


__all__ = ["SharpaHand"]
