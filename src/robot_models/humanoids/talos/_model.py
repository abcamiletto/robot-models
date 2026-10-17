"""PAL Robotics TALOS humanoid."""

from types import MappingProxyType

from robot_models._base import Humanoid
from robot_models._constants import Joint


class Talos(Humanoid):
    """Rigid articulated PAL Robotics TALOS model, with its two grippers."""

    _NAME = "talos"
    _COMMON_JOINTS = MappingProxyType(
        {
            Joint.PELVIS: "base_link",
            Joint.NECK: "head_1_link",
            Joint.HEAD: "head_2_link",
            Joint.LEFT_SHOULDER: "arm_left_2_link",
            Joint.RIGHT_SHOULDER: "arm_right_2_link",
            Joint.LEFT_ELBOW: "arm_left_4_link",
            Joint.RIGHT_ELBOW: "arm_right_4_link",
            Joint.LEFT_WRIST: "arm_left_5_link",
            Joint.RIGHT_WRIST: "arm_right_5_link",
            Joint.LEFT_HIP: "leg_left_1_link",
            Joint.RIGHT_HIP: "leg_right_1_link",
            Joint.LEFT_KNEE: "leg_left_4_link",
            Joint.RIGHT_KNEE: "leg_right_4_link",
            Joint.LEFT_ANKLE: "leg_left_5_link",
            Joint.RIGHT_ANKLE: "leg_right_5_link",
            Joint.LEFT_FOOT: "leg_left_6_link",
            Joint.RIGHT_FOOT: "leg_right_6_link",
        }
    )


__all__ = ["Talos"]
