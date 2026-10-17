"""PSYONIC Ability Hand."""

from types import MappingProxyType

from robot_models._base import Hand


class AbilityHand(Hand):
    """Rigid articulated PSYONIC Ability Hand; each finger's PIP follows its MCP."""

    _NAME = "ability"
    _HAND_JOINTS = MappingProxyType(
        {
            "wrist": "{side}_base_link",
            "thumb_cmc": "{side}_thumb_metacarpal_link",
            "thumb_mcp": "{side}_thumb_proximal_link",
            "index_mcp": "{side}_index_proximal_link",
            "index_pip": "{side}_index_distal_link",
            "middle_mcp": "{side}_middle_proximal_link",
            "middle_pip": "{side}_middle_distal_link",
            "ring_mcp": "{side}_ring_proximal_link",
            "ring_pip": "{side}_ring_distal_link",
            "pinky_mcp": "{side}_pinky_proximal_link",
            "pinky_pip": "{side}_pinky_distal_link",
        }
    )


__all__ = ["AbilityHand"]
