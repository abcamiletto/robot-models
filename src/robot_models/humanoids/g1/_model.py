"""Unitree G1 humanoid."""

from types import MappingProxyType

from robot_models._base import Humanoid
from robot_models._constants import Joint


class G1(Humanoid):
    """Rigid articulated Unitree G1 model (29-DoF body)."""

    _NAME = "g1"
    _COMMON_JOINTS = MappingProxyType(
        {
            Joint.PELVIS: "pelvis",
            Joint.LEFT_SHOULDER: "left_shoulder_pitch_link",
            Joint.RIGHT_SHOULDER: "right_shoulder_pitch_link",
            Joint.LEFT_ELBOW: "left_elbow_link",
            Joint.RIGHT_ELBOW: "right_elbow_link",
            Joint.LEFT_WRIST: "left_wrist_roll_link",
            Joint.RIGHT_WRIST: "right_wrist_roll_link",
            Joint.LEFT_HIP: "left_hip_pitch_link",
            Joint.RIGHT_HIP: "right_hip_pitch_link",
            Joint.LEFT_KNEE: "left_knee_link",
            Joint.RIGHT_KNEE: "right_knee_link",
            Joint.LEFT_ANKLE: "left_ankle_pitch_link",
            Joint.RIGHT_ANKLE: "right_ankle_pitch_link",
            Joint.LEFT_FOOT: "left_ankle_roll_link",
            Joint.RIGHT_FOOT: "right_ankle_roll_link",
        }
    )


__all__ = ["G1"]
