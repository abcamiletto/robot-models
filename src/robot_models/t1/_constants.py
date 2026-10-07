from robot_models._common.pose_assets import load_npz
from robot_models._constants import Joint

T1_JOINTS = {
    Joint.PELVIS: "waist_skel",
    Joint.NECK: "head_yaw_skel",
    Joint.HEAD: "head_pitch_skel",
    Joint.LEFT_SHOULDER: "left_shoulder_pitch_skel",
    Joint.RIGHT_SHOULDER: "right_shoulder_pitch_skel",
    Joint.LEFT_ELBOW: "left_elbow_yaw_skel",
    Joint.RIGHT_ELBOW: "right_elbow_yaw_skel",
    Joint.LEFT_HIP: "left_hip_pitch_skel",
    Joint.RIGHT_HIP: "right_hip_pitch_skel",
    Joint.LEFT_KNEE: "left_knee_pitch_skel",
    Joint.RIGHT_KNEE: "right_knee_pitch_skel",
    Joint.LEFT_ANKLE: "left_ankle_pitch_skel",
    Joint.RIGHT_ANKLE: "right_ankle_pitch_skel",
}


_POSES = load_npz("robot_models.t1")

# Hinge angles in actuated joint order, so presets do not depend on the convention.
T1_BODY_PRESETS = _POSES["body"]

__all__ = ["T1_BODY_PRESETS", "T1_JOINTS"]
