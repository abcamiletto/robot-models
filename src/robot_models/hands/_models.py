"""Canonical model data; geometry and kinematics come from the source assets."""

from types import MappingProxyType

from robot_models._spec import ModelSpec

MODELS = {
    "ability": ModelSpec(
        class_name="AbilityHand",
        family="hands",
        joints=MappingProxyType(
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
        ),
        sides=("right", "left"),
    ),
    "allegro": ModelSpec(
        class_name="AllegroHand",
        family="hands",
        joints=MappingProxyType(
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
        ),
        sides=("right", "left"),
    ),
    "barrett": ModelSpec(
        class_name="BarrettHand",
        family="hands",
        joints=MappingProxyType({"wrist": "bh_base_link"}),
        sides=("right",),
        asset_version=3,
    ),
    "brainco": ModelSpec(
        class_name="BrainCoHand",
        family="hands",
        joints=MappingProxyType(
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
        ),
        sides=("right", "left"),
    ),
    "ezgripper": ModelSpec(
        class_name="EZGripper",
        family="hands",
        joints=MappingProxyType({"wrist": "mount"}),
        sides=("right",),
    ),
    "inspire": ModelSpec(
        class_name="InspireHand",
        family="hands",
        joints=MappingProxyType(
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
        ),
        sides=("right", "left"),
    ),
    # Source body names use bs/px/md/ds for base, proximal, middle and distal.
    "leap": ModelSpec(
        class_name="LeapHand",
        family="hands",
        joints=MappingProxyType(
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
        ),
        sides=("right", "left"),
    ),
    "robotiq-3finger": ModelSpec(
        class_name="Robotiq3Finger",
        family="hands",
        joints=MappingProxyType({"wrist": "palm"}),
        sides=("right",),
        asset_version=3,
    ),
    "sharpa": ModelSpec(
        class_name="SharpaHand",
        family="hands",
        joints=MappingProxyType(
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
        ),
        sides=("right", "left"),
    ),
    # Anatomical joints with coincident hinges use the distal body of each pair.
    "shadow": ModelSpec(
        class_name="ShadowHand",
        family="hands",
        joints=MappingProxyType(
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
        ),
        sides=("right", "left"),
        asset_version=2,
    ),
}
