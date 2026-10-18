"""Canonical model data; geometry and kinematics come from the source assets."""

from math import inf
from types import MappingProxyType

from robot_models._spec import Dof, Finger, ModelSpec

# Nominal maximal voluntary torques (N·m) of MANO's bend and spread hinges (thumb CMC: and twist), per finger
# joint: fingertip strength (40-50 N for the index) times each joint's lever, weaker for the ring and pinky.
# Hinges no muscle turns are held by the joint's passive structure: unbounded.
_MANO_EFFORT: dict[Finger, tuple[tuple[float, ...], ...]] = {
    "index": ((3.0, 0.8), (1.5,), (0.7,)),
    "middle": ((3.0, 0.8), (1.5,), (0.7,)),
    "pinky": ((1.5, 0.5), (0.8,), (0.4,)),
    "ring": ((2.0, 0.6), (1.0,), (0.5,)),
    "thumb": ((3.0, 2.0, 1.0), (2.0,), (1.0,)),
}


def _mano_dofs(finger: Finger, joint: int, bend: float, spread: float = inf, twist: float = inf) -> dict[str, Dof]:
    """A MANO joint's three hinges: bend flexes, spread abducts, twist rotates the finger."""
    return {
        f"{finger}{joint}_bend": Dof(finger, "flexion", 1, bend),
        f"{finger}{joint}_spread": Dof(finger, "abduction", 0, spread),
        f"{finger}{joint}_twist": Dof(finger, "rotation", 0, twist),
    }


MODELS = {
    "ability": ModelSpec(
        class_name="AbilityHand",
        dofs=MappingProxyType(
            {
                "{side}_thumb_cmc_joint": Dof("thumb", "rotation", 0),
                "{side}_thumb_mcp_joint": Dof("thumb", "flexion", 1),
                **{
                    f"{{side}}_{finger}_mcp_joint": Dof(finger, "flexion", 1)
                    for finger in ("index", "middle", "ring", "pinky")
                },
            }
        ),
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
        dofs=MappingProxyType(
            {
                **{
                    f"{prefix}j{index}": Dof(
                        finger, "abduction" if index == 0 else "flexion", 0 if index == 0 else 1, 0.7
                    )
                    for prefix, finger in (("ff", "index"), ("mf", "middle"), ("rf", "ring"))
                    for index in range(4)
                },
                "thj0": Dof("thumb", "rotation", 0, 0.7),
                "thj1": Dof("thumb", "abduction", 0, 0.7),
                "thj2": Dof("thumb", "flexion", 1, 0.7),
                "thj3": Dof("thumb", "flexion", 1, 0.7),
            }
        ),
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
        dofs=MappingProxyType(
            {
                "bh_j32_joint": Dof("thumb", "flexion", 1),
                "bh_j11_joint": Dof("palm", "abduction", 0),
                "bh_j12_joint": Dof("index", "flexion", 1),
                "bh_j22_joint": Dof("middle", "flexion", 1),
            }
        ),
        family="hands",
        joints=MappingProxyType({"wrist": "bh_base_link"}),
        sides=("right",),
        asset_version=3,
    ),
    "brainco": ModelSpec(
        class_name="BrainCoHand",
        dofs=MappingProxyType(
            {
                "{side}_thumb_metacarpal_joint": Dof("thumb", "abduction", 0),
                "{side}_thumb_proximal_joint": Dof("thumb", "flexion", 1),
                **{
                    f"{{side}}_{finger}_proximal_joint": Dof(finger, "flexion", 1)
                    for finger in ("index", "middle", "ring", "pinky")
                },
            }
        ),
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
        dofs=MappingProxyType(
            {
                f"F{index}_{joint}": Dof(finger, "flexion", 1)
                for index, finger in ((1, "thumb"), (2, "index"))
                for joint in ("palm_knuckle", "knuckle_tip")
            }
        ),
        family="hands",
        joints=MappingProxyType({"wrist": "mount"}),
        sides=("right",),
    ),
    "inspire": ModelSpec(
        class_name="InspireHand",
        dofs=MappingProxyType(
            {
                "{side}_thumb_proximal_yaw_joint": Dof("thumb", "rotation", 0),
                "{side}_thumb_proximal_pitch_joint": Dof("thumb", "flexion", 1),
                **{
                    f"{{side}}_{finger}_proximal_joint": Dof(finger, "flexion", 1)
                    for finger in ("index", "middle", "ring", "pinky")
                },
            }
        ),
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
        dofs=MappingProxyType(
            {
                **{
                    f"{prefix}_{joint}": Dof(
                        finger, "abduction" if joint == "rot" else "flexion", 0 if joint == "rot" else 1, 0.93
                    )
                    for prefix, finger in (("if", "index"), ("mf", "middle"), ("rf", "ring"))
                    for joint in ("mcp", "rot", "pip", "dip")
                },
                "th_cmc": Dof("thumb", "abduction", 0, 0.93),
                "th_axl": Dof("thumb", "rotation", 0, 0.93),
                "th_mcp": Dof("thumb", "flexion", 1, 0.93),
                "th_ipl": Dof("thumb", "flexion", 1, 0.93),
            }
        ),
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
    # Built locally from the user's MANO files (Body Models); three hinges per joint: bend, spread, twist.
    "mano": ModelSpec(
        class_name="ManoRobot",
        dofs=MappingProxyType(
            {
                name: dof
                for finger, joints in _MANO_EFFORT.items()
                for joint, efforts in enumerate(joints, 1)
                for name, dof in _mano_dofs(finger, joint, *efforts).items()
            }
        ),
        family="hands",
        joints=MappingProxyType(
            {
                "wrist": "wrist",
                **{
                    f"{finger}_{joint}": f"{finger}{index + 1}"
                    for finger in ("index", "middle", "ring", "pinky")
                    for index, joint in enumerate(("mcp", "pip", "dip"))
                },
                **{f"thumb_{joint}": f"thumb{index + 1}" for index, joint in enumerate(("cmc", "mcp", "ip"))},
            }
        ),
        sides=("right", "left"),
        body_model="mano",
    ),
    "robotiq-3finger": ModelSpec(
        class_name="Robotiq3Finger",
        dofs=MappingProxyType(
            {
                **{f"palm_finger_{index}_joint": Dof("palm", "abduction", 0) for index in (1, 2)},
                **{
                    f"finger_{index}_joint_{joint}": Dof(finger, "flexion", 1)
                    for index, finger in (("1", "index"), ("2", "middle"), ("middle", "thumb"))
                    for joint in (1, 2, 3)
                },
            }
        ),
        family="hands",
        joints=MappingProxyType({"wrist": "palm"}),
        sides=("right",),
        asset_version=3,
    ),
    "sharpa": ModelSpec(
        class_name="SharpaHand",
        dofs=MappingProxyType(
            {
                **{
                    f"{{side}}_{finger}_{joint}": Dof(
                        finger, "abduction" if joint.endswith("AA") else "flexion", 0 if joint.endswith("AA") else 1
                    )
                    for finger in ("thumb", "index", "middle", "ring", "pinky")
                    for joint in (
                        ("CMC_FE", "CMC_AA", "MCP_FE", "MCP_AA", "IP")
                        if finger == "thumb"
                        else ("MCP_FE", "MCP_AA", "PIP", "DIP")
                    )
                },
                "{side}_pinky_CMC": Dof("palm", "flexion", 1),
            }
        ),
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
        dofs=MappingProxyType(
            {
                "{s}h_WRJ2": Dof("wrist", "abduction", 0),
                "{s}h_WRJ1": Dof("wrist", "flexion", 0),
                **{
                    f"{{s}}h_{prefix}J{index}": Dof(
                        finger, "abduction" if index == 4 else "flexion", 0 if index == 4 else 1
                    )
                    for prefix, finger in (("FF", "index"), ("MF", "middle"), ("RF", "ring"), ("LF", "pinky"))
                    for index in (4, 3, 2, 1)
                },
                "{s}h_LFJ5": Dof("palm", "flexion", 1),
                "{s}h_THJ5": Dof("thumb", "rotation", 0),
                "{s}h_THJ4": Dof("thumb", "abduction", 0),
                "{s}h_THJ3": Dof("thumb", "abduction", 0),
                "{s}h_THJ2": Dof("thumb", "flexion", 1),
                "{s}h_THJ1": Dof("thumb", "flexion", 1),
            }
        ),
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
