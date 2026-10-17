"""Booster Robotics T1 humanoid."""

from types import MappingProxyType

from robot_models._base import Humanoid
from robot_models._constants import Joint


class T1(Humanoid):
    """Rigid articulated Booster T1 model."""

    _NAME = "t1"
    # Upstream body names: AL*/AR* are the arm links, H1/H2 the head links.
    _MIRROR = (("left", "right"), ("Left", "Right"), ("AL", "AR"))
    # The upstream Elbow_Yaw hinge, in *_hand_link, is the elbow bend.
    _COMMON_JOINTS = MappingProxyType(
        {
            Joint.PELVIS: "Waist",
            Joint.NECK: "H1",
            Joint.HEAD: "H2",
            Joint.LEFT_SHOULDER: "AL1",
            Joint.RIGHT_SHOULDER: "AR1",
            Joint.LEFT_ELBOW: "left_hand_link",
            Joint.RIGHT_ELBOW: "right_hand_link",
            Joint.LEFT_HIP: "Hip_Pitch_Left",
            Joint.RIGHT_HIP: "Hip_Pitch_Right",
            Joint.LEFT_KNEE: "Shank_Left",
            Joint.RIGHT_KNEE: "Shank_Right",
            Joint.LEFT_ANKLE: "Ankle_Cross_Left",
            Joint.RIGHT_ANKLE: "Ankle_Cross_Right",
            Joint.LEFT_FOOT: "left_foot_link",
            Joint.RIGHT_FOOT: "right_foot_link",
        }
    )


__all__ = ["T1"]
