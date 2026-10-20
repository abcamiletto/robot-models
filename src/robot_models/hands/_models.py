"""Canonical model data; geometry and kinematics come from the source assets."""

from types import MappingProxyType
from typing import cast

from robot_models._skinned._sources import MANO_HINGES, MANO_ROLES, SOURCES
from robot_models._spec import Dof, Finger, ModelSpec

MODELS = {
    "ability": ModelSpec(
        class_name="AbilityHand",
        dofs=MappingProxyType(
            {
                "{side}_thumb_cmc_joint": Dof("thumb", "rotation"),
                "{side}_thumb_mcp_joint": Dof("thumb", "flexion"),
                **{
                    f"{{side}}_{finger}_mcp_joint": Dof(finger, "flexion")
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
        variants=("right", "left"),
    ),
    "allegro": ModelSpec(
        class_name="AllegroHand",
        dofs=MappingProxyType(
            {
                **{
                    f"{prefix}j{index}": Dof(finger, "abduction" if index == 0 else "flexion", 0.7)
                    for prefix, finger in (("ff", "index"), ("mf", "middle"), ("rf", "ring"))
                    for index in range(4)
                },
                "thj0": Dof("thumb", "rotation", 0.7),
                "thj1": Dof("thumb", "abduction", 0.7),
                "thj2": Dof("thumb", "flexion", 0.7),
                "thj3": Dof("thumb", "flexion", 0.7),
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
        variants=("right", "left"),
    ),
    "barrett": ModelSpec(
        class_name="BarrettHand",
        dofs=MappingProxyType(
            {
                "bh_j32_joint": Dof("thumb", "flexion"),
                "bh_j11_joint": Dof("palm", "abduction"),
                "bh_j12_joint": Dof("index", "flexion"),
                "bh_j22_joint": Dof("middle", "flexion"),
            }
        ),
        family="hands",
        joints=MappingProxyType({"wrist": "bh_base_link"}),
        variants=("right",),
        asset_version=3,
    ),
    "brainco": ModelSpec(
        class_name="BrainCoHand",
        dofs=MappingProxyType(
            {
                "{side}_thumb_metacarpal_joint": Dof("thumb", "abduction"),
                "{side}_thumb_proximal_joint": Dof("thumb", "flexion"),
                **{
                    f"{{side}}_{finger}_proximal_joint": Dof(finger, "flexion")
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
        variants=("right", "left"),
    ),
    "ezgripper": ModelSpec(
        class_name="EZGripper",
        dofs=MappingProxyType(
            {
                f"F{index}_{joint}": Dof(finger, "flexion")
                for index, finger in ((1, "thumb"), (2, "index"))
                for joint in ("palm_knuckle", "knuckle_tip")
            }
        ),
        family="hands",
        joints=MappingProxyType({"wrist": "mount"}),
        variants=("right",),
    ),
    "inspire": ModelSpec(
        class_name="InspireHand",
        dofs=MappingProxyType(
            {
                "{side}_thumb_proximal_yaw_joint": Dof("thumb", "rotation"),
                "{side}_thumb_proximal_pitch_joint": Dof("thumb", "flexion"),
                **{
                    f"{{side}}_{finger}_proximal_joint": Dof(finger, "flexion")
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
        variants=("right", "left"),
    ),
    # Source body names use bs/px/md/ds for base, proximal, middle and distal.
    "leap": ModelSpec(
        class_name="LeapHand",
        dofs=MappingProxyType(
            {
                **{
                    f"{prefix}_{joint}": Dof(finger, "abduction" if joint == "rot" else "flexion", 0.93)
                    for prefix, finger in (("if", "index"), ("mf", "middle"), ("rf", "ring"))
                    for joint in ("mcp", "rot", "pip", "dip")
                },
                "th_cmc": Dof("thumb", "abduction", 0.93),
                "th_axl": Dof("thumb", "rotation", 0.93),
                "th_mcp": Dof("thumb", "flexion", 0.93),
                "th_ipl": Dof("thumb", "flexion", 0.93),
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
        variants=("right", "left"),
    ),
    # Built locally from the user's MANO files (Body Models); three hinges per joint: bend, spread, twist.
    "mano": ModelSpec(
        class_name="ManoRobot",
        dofs=MappingProxyType(
            {
                f"{joint}_{hinge}": Dof(cast(Finger, joint[:-1]), role, effort)
                for joint, hinges in MANO_HINGES.items()
                for (hinge, role), (effort, _, _) in zip(MANO_ROLES, hinges, strict=True)
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
        variants=("right", "left"),
        source=SOURCES["mano"],
    ),
    "robotiq-3finger": ModelSpec(
        class_name="Robotiq3Finger",
        dofs=MappingProxyType(
            {
                **{f"palm_finger_{index}_joint": Dof("palm", "abduction") for index in (1, 2)},
                **{
                    f"finger_{index}_joint_{joint}": Dof(finger, "flexion")
                    for index, finger in (("1", "index"), ("2", "middle"), ("middle", "thumb"))
                    for joint in (1, 2, 3)
                },
            }
        ),
        family="hands",
        joints=MappingProxyType({"wrist": "palm"}),
        variants=("right",),
        asset_version=3,
    ),
    "sharpa": ModelSpec(
        class_name="SharpaHand",
        dofs=MappingProxyType(
            {
                **{
                    f"{{side}}_{finger}_{joint}": Dof(finger, "abduction" if joint.endswith("AA") else "flexion")
                    for finger in ("thumb", "index", "middle", "ring", "pinky")
                    for joint in (
                        ("CMC_FE", "CMC_AA", "MCP_FE", "MCP_AA", "IP")
                        if finger == "thumb"
                        else ("MCP_FE", "MCP_AA", "PIP", "DIP")
                    )
                },
                "{side}_pinky_CMC": Dof("palm", "flexion"),
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
        variants=("right", "left"),
    ),
    # Anatomical joints with coincident hinges use the distal body of each pair.
    "shadow": ModelSpec(
        class_name="ShadowHand",
        dofs=MappingProxyType(
            {
                "{s}h_WRJ2": Dof("wrist", "abduction"),
                "{s}h_WRJ1": Dof("wrist", "flexion"),
                **{
                    f"{{s}}h_{prefix}J{index}": Dof(finger, "abduction" if index == 4 else "flexion")
                    for prefix, finger in (("FF", "index"), ("MF", "middle"), ("RF", "ring"), ("LF", "pinky"))
                    for index in (4, 3, 2, 1)
                },
                "{s}h_LFJ5": Dof("palm", "flexion"),
                "{s}h_THJ5": Dof("thumb", "rotation"),
                "{s}h_THJ4": Dof("thumb", "abduction"),
                "{s}h_THJ3": Dof("thumb", "abduction"),
                "{s}h_THJ2": Dof("thumb", "flexion"),
                "{s}h_THJ1": Dof("thumb", "flexion"),
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
        variants=("right", "left"),
        asset_version=3,
        # OpenAI Gym's (DexGraspNet's): its little-finger metacarpal hinge and thumb base sit elsewhere in the palm.
        kinematics=MappingProxyType({"openai": ("right",)}),
    ),
}
