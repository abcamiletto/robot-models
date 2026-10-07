from robot_models._common.pose_assets import load_npz
from robot_models._constants import Joint

TALOS_JOINTS = {
    Joint.PELVIS: "base_skel",
    Joint.NECK: "head_1_skel",
    Joint.HEAD: "head",
    Joint.LEFT_SHOULDER: "left_arm_2_skel",
    Joint.RIGHT_SHOULDER: "right_arm_2_skel",
    Joint.LEFT_ELBOW: "left_arm_4_skel",
    Joint.RIGHT_ELBOW: "right_arm_4_skel",
    Joint.LEFT_WRIST: "left_arm_5_skel",
    Joint.RIGHT_WRIST: "right_arm_5_skel",
    Joint.LEFT_HIP: "left_leg_1_skel",
    Joint.RIGHT_HIP: "right_leg_1_skel",
    Joint.LEFT_KNEE: "left_leg_4_skel",
    Joint.RIGHT_KNEE: "right_leg_4_skel",
    Joint.LEFT_ANKLE: "left_leg_5_skel",
    Joint.RIGHT_ANKLE: "right_leg_5_skel",
    Joint.LEFT_FOOT: "left_foot",
    Joint.RIGHT_FOOT: "right_foot",
}


_POSES = load_npz("robot_models.talos")

TALOS_BODY_PRESETS = _POSES["body"]

__all__ = ["TALOS_BODY_PRESETS", "TALOS_JOINTS"]
