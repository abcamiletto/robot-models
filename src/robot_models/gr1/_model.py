"""Fourier GR1 humanoid."""

from types import MappingProxyType

from robot_models._base import Humanoid
from robot_models._constants import Joint


class GR1(Humanoid):
    """Rigid articulated Fourier GR1T2 model, without dexterous hands."""

    _NAME = "gr1"
    _COMMON_JOINTS = MappingProxyType(
        {
            Joint.PELVIS: "base_link",
            Joint.NECK: "head_roll_link",
            Joint.HEAD: "head_yaw_link",
            Joint.LEFT_SHOULDER: "left_upper_arm_pitch_link",
            Joint.RIGHT_SHOULDER: "right_upper_arm_pitch_link",
            Joint.LEFT_ELBOW: "left_lower_arm_pitch_link",
            Joint.RIGHT_ELBOW: "right_lower_arm_pitch_link",
            Joint.LEFT_WRIST: "left_hand_roll_link",
            Joint.RIGHT_WRIST: "right_hand_roll_link",
            Joint.LEFT_HIP: "left_thigh_pitch_link",
            Joint.RIGHT_HIP: "right_thigh_pitch_link",
            Joint.LEFT_KNEE: "left_shank_pitch_link",
            Joint.RIGHT_KNEE: "right_shank_pitch_link",
            Joint.LEFT_ANKLE: "left_foot_pitch_link",
            Joint.RIGHT_ANKLE: "right_foot_pitch_link",
            Joint.LEFT_FOOT: "left_foot_roll_link",
            Joint.RIGHT_FOOT: "right_foot_roll_link",
        }
    )


__all__ = ["GR1"]
