"""Inspire RH56 hand."""

from types import MappingProxyType

from robot_models._base import Hand


class InspireHand(Hand):
    """Rigid articulated Inspire RH56 hand; intermediate and distal joints follow their drivers."""

    _NAME = "inspire"
    _HAND_JOINTS = MappingProxyType(
        {
            "wrist": "{side}_hand_base_link",
            "thumb_cmc": "{side}_thumb_proximal_base",
            "thumb_mcp": "{side}_thumb_intermediate",
            "thumb_ip": "{side}_thumb_distal",
            "index_mcp": "{side}_index_proximal",
            "index_pip": "{side}_index_intermediate",
            "middle_mcp": "{side}_middle_proximal",
            "middle_pip": "{side}_middle_intermediate",
            "ring_mcp": "{side}_ring_proximal",
            "ring_pip": "{side}_ring_intermediate",
            "pinky_mcp": "{side}_pinky_proximal",
            "pinky_pip": "{side}_pinky_intermediate",
        }
    )


__all__ = ["InspireHand"]
