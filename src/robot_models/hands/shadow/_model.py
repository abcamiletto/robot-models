"""Shadow Dexterous Hand."""

from types import MappingProxyType

from robot_models._base import Hand


class ShadowHand(Hand):
    """Rigid articulated Shadow Dexterous Hand E3M5, from the forearm."""

    _NAME = "shadow"
    # Two-DoF anatomical joints map to the distal body of each coincident pair.
    _HAND_JOINTS = MappingProxyType(
        {
            "wrist": "{s}h_palm",
            "thumb_cmc": "{s}h_thproximal",
            "thumb_mcp": "{s}h_thmiddle",
            "thumb_ip": "{s}h_thdistal",
            "index_mcp": "{s}h_ffproximal",
            "index_pip": "{s}h_ffmiddle",
            "index_dip": "{s}h_ffdistal",
            "middle_mcp": "{s}h_mfproximal",
            "middle_pip": "{s}h_mfmiddle",
            "middle_dip": "{s}h_mfdistal",
            "ring_mcp": "{s}h_rfproximal",
            "ring_pip": "{s}h_rfmiddle",
            "ring_dip": "{s}h_rfdistal",
            "pinky_mcp": "{s}h_lfproximal",
            "pinky_pip": "{s}h_lfmiddle",
            "pinky_dip": "{s}h_lfdistal",
        }
    )


__all__ = ["ShadowHand"]
