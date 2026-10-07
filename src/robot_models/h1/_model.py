"""Unitree H1 humanoid."""

from types import MappingProxyType

from robot_models._base import Humanoid
from robot_models._constants import Joint


class H1(Humanoid):
    """Rigid articulated Unitree H1 model (original 19-DoF H1)."""

    _NAME = "h1"
    # H1 has no wrist or separate foot bodies. The hip roll origin is where the
    # hip roll and pitch axes intersect.
    _COMMON_JOINTS = MappingProxyType(
        {
            Joint.PELVIS: "pelvis",
            Joint.LEFT_SHOULDER: "left_shoulder_pitch_link",
            Joint.RIGHT_SHOULDER: "right_shoulder_pitch_link",
            Joint.LEFT_ELBOW: "left_elbow_link",
            Joint.RIGHT_ELBOW: "right_elbow_link",
            Joint.LEFT_HIP: "left_hip_roll_link",
            Joint.RIGHT_HIP: "right_hip_roll_link",
            Joint.LEFT_KNEE: "left_knee_link",
            Joint.RIGHT_KNEE: "right_knee_link",
            Joint.LEFT_ANKLE: "left_ankle_link",
            Joint.RIGHT_ANKLE: "right_ankle_link",
        }
    )


__all__ = ["H1"]
