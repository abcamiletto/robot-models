"""Wonik Robotics Allegro Hand."""

from types import MappingProxyType

from robot_models._base import Hand


class AllegroHand(Hand):
    """Rigid articulated Allegro Hand V3 with four fingers."""

    _NAME = "allegro"
    _HAND_JOINTS = MappingProxyType(
        {
            "wrist": "palm",
            "thumb_cmc": "th_base",
            "thumb_mcp": "th_medial",
            "thumb_ip": "th_distal",
            "index_mcp": "ff_proximal",
            "index_pip": "ff_medial",
            "index_dip": "ff_distal",
            "middle_mcp": "mf_proximal",
            "middle_pip": "mf_medial",
            "middle_dip": "mf_distal",
            "ring_mcp": "rf_proximal",
            "ring_pip": "rf_medial",
            "ring_dip": "rf_distal",
        }
    )


__all__ = ["AllegroHand"]
