"""Canonical model data; geometry and kinematics come from the source assets."""

from types import MappingProxyType

from robot_models._spec import ModelSpec

MODELS = {
    "g1": ModelSpec(
        class_name="G1",
        family="humanoids",
        mirror=(("left", "right"),),
        joints=MappingProxyType(
            {
                "pelvis": "pelvis",
                "left_shoulder": "left_shoulder_pitch_link",
                "right_shoulder": "right_shoulder_pitch_link",
                "left_elbow": "left_elbow_link",
                "right_elbow": "right_elbow_link",
                "left_wrist": "left_wrist_roll_link",
                "right_wrist": "right_wrist_roll_link",
                "left_hip": "left_hip_pitch_link",
                "right_hip": "right_hip_pitch_link",
                "left_knee": "left_knee_link",
                "right_knee": "right_knee_link",
                "left_ankle": "left_ankle_pitch_link",
                "right_ankle": "right_ankle_pitch_link",
                "left_foot": "left_ankle_roll_link",
                "right_foot": "right_ankle_roll_link",
            }
        ),
    ),
    "gr1": ModelSpec(
        class_name="GR1",
        family="humanoids",
        mirror=(("left", "right"),),
        joints=MappingProxyType(
            {
                "pelvis": "base_link",
                "neck": "head_roll_link",
                "head": "head_yaw_link",
                "left_shoulder": "left_upper_arm_pitch_link",
                "right_shoulder": "right_upper_arm_pitch_link",
                "left_elbow": "left_lower_arm_pitch_link",
                "right_elbow": "right_lower_arm_pitch_link",
                "left_wrist": "left_hand_roll_link",
                "right_wrist": "right_hand_roll_link",
                "left_hip": "left_thigh_pitch_link",
                "right_hip": "right_thigh_pitch_link",
                "left_knee": "left_shank_pitch_link",
                "right_knee": "right_shank_pitch_link",
                "left_ankle": "left_foot_pitch_link",
                "right_ankle": "right_foot_pitch_link",
                "left_foot": "left_foot_roll_link",
                "right_foot": "right_foot_roll_link",
            }
        ),
    ),
    "h1": ModelSpec(
        class_name="H1",
        family="humanoids",
        mirror=(("left", "right"),),
        joints=MappingProxyType(
            {
                "pelvis": "pelvis",
                "left_shoulder": "left_shoulder_pitch_link",
                "right_shoulder": "right_shoulder_pitch_link",
                "left_elbow": "left_elbow_link",
                "right_elbow": "right_elbow_link",
                "left_hip": "left_hip_roll_link",
                "right_hip": "right_hip_roll_link",
                "left_knee": "left_knee_link",
                "right_knee": "right_knee_link",
                "left_ankle": "left_ankle_link",
                "right_ankle": "right_ankle_link",
            }
        ),
    ),
    # Built locally from the user's SMPL files (Body Models); three hinges per joint, about x, y and z.
    "smpl": ModelSpec(
        class_name="SmplRobot",
        family="humanoids",
        mirror=(("left", "right"),),
        joints=MappingProxyType(
            {
                **{joint: joint for joint in ("pelvis", "neck", "head")},
                **{
                    f"{side}_{joint}": f"{side}_{joint}"
                    for side in ("left", "right")
                    for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "foot")
                },
            }
        ),
        body_model="smpl",
    ),
    "t1": ModelSpec(
        class_name="T1",
        family="humanoids",
        mirror=(("left", "right"), ("Left", "Right"), ("AL", "AR")),
        joints=MappingProxyType(
            {
                "pelvis": "Waist",
                "neck": "H1",
                "head": "H2",
                "left_shoulder": "AL1",
                "right_shoulder": "AR1",
                "left_elbow": "left_hand_link",
                "right_elbow": "right_hand_link",
                "left_hip": "Hip_Pitch_Left",
                "right_hip": "Hip_Pitch_Right",
                "left_knee": "Shank_Left",
                "right_knee": "Shank_Right",
                "left_ankle": "Ankle_Cross_Left",
                "right_ankle": "Ankle_Cross_Right",
                "left_foot": "left_foot_link",
                "right_foot": "right_foot_link",
            }
        ),
    ),
    "talos": ModelSpec(
        class_name="Talos",
        family="humanoids",
        mirror=(("left", "right"),),
        joints=MappingProxyType(
            {
                "pelvis": "base_link",
                "neck": "head_1_link",
                "head": "head_2_link",
                "left_shoulder": "arm_left_2_link",
                "right_shoulder": "arm_right_2_link",
                "left_elbow": "arm_left_4_link",
                "right_elbow": "arm_right_4_link",
                "left_wrist": "arm_left_5_link",
                "right_wrist": "arm_right_5_link",
                "left_hip": "leg_left_1_link",
                "right_hip": "leg_right_1_link",
                "left_knee": "leg_left_4_link",
                "right_knee": "leg_right_4_link",
                "left_ankle": "leg_left_5_link",
                "right_ankle": "leg_right_5_link",
                "left_foot": "leg_left_6_link",
                "right_foot": "leg_right_6_link",
            }
        ),
    ),
}
