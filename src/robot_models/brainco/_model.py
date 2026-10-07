"""BrainCo Revo 2 hand."""

from types import MappingProxyType

from robot_models._base import Hand


class BrainCoHand(Hand):
    """Rigid articulated BrainCo Revo 2 hand; each distal joint follows its proximal joint."""

    _NAME = "brainco"
    _HAND_JOINTS = MappingProxyType(
        {
            "wrist": "{side}_base_link",
            "thumb_cmc": "{side}_thumb_metacarpal_link",
            "thumb_mcp": "{side}_thumb_proximal_link",
            "thumb_ip": "{side}_thumb_distal_link",
            "index_mcp": "{side}_index_proximal_link",
            "index_dip": "{side}_index_distal_link",
            "middle_mcp": "{side}_middle_proximal_link",
            "middle_dip": "{side}_middle_distal_link",
            "ring_mcp": "{side}_ring_proximal_link",
            "ring_dip": "{side}_ring_distal_link",
            "pinky_mcp": "{side}_pinky_proximal_link",
            "pinky_dip": "{side}_pinky_distal_link",
        }
    )


__all__ = ["BrainCoHand"]
