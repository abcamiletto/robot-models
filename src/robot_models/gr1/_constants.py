from robot_models._common.pose_assets import load_npz
from robot_models._constants import Joint

GR1_JOINTS = {
    Joint.PELVIS: "base_skel",
    Joint.NECK: "head_roll_skel",
    Joint.HEAD: "head_yaw_skel",
    Joint.LEFT_SHOULDER: "left_shoulder_pitch_skel",
    Joint.RIGHT_SHOULDER: "right_shoulder_pitch_skel",
    Joint.LEFT_ELBOW: "left_elbow_pitch_skel",
    Joint.RIGHT_ELBOW: "right_elbow_pitch_skel",
    Joint.LEFT_WRIST: "left_wrist_roll_skel",
    Joint.RIGHT_WRIST: "right_wrist_roll_skel",
    Joint.LEFT_HIP: "left_hip_pitch_skel",
    Joint.RIGHT_HIP: "right_hip_pitch_skel",
    Joint.LEFT_KNEE: "left_knee_pitch_skel",
    Joint.RIGHT_KNEE: "right_knee_pitch_skel",
    Joint.LEFT_ANKLE: "left_ankle_pitch_skel",
    Joint.RIGHT_ANKLE: "right_ankle_pitch_skel",
}


_POSES = load_npz("robot_models.gr1")

GR1_BODY_PRESETS = _POSES["body"]

__all__ = ["GR1_BODY_PRESETS", "GR1_JOINTS"]
