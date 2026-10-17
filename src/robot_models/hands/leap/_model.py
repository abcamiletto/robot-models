"""LEAP Hand."""

from types import MappingProxyType

from robot_models._base import Hand


class LeapHand(Hand):
    """Rigid articulated LEAP Hand v1 with four fingers."""

    _NAME = "leap"
    # Upstream body names: bs/px/md/ds are base, proximal, middle, and distal links.
    _HAND_JOINTS = MappingProxyType(
        {
            "wrist": "palm",
            "thumb_cmc": "th_mp",
            "thumb_mcp": "th_px",
            "thumb_ip": "th_ds",
            "index_mcp": "if_bs",
            "index_pip": "if_md",
            "index_dip": "if_ds",
            "middle_mcp": "mf_bs",
            "middle_pip": "mf_md",
            "middle_dip": "mf_ds",
            "ring_mcp": "rf_bs",
            "ring_pip": "rf_md",
            "ring_dip": "rf_ds",
        }
    )


__all__ = ["LeapHand"]
