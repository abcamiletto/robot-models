from robot_models._common.pose_assets import load_npz
from robot_models._constants import Joint

# Two-DoF anatomical joints map to the distal body of each coincident pair.
LEFT_SHADOW_JOINTS = {
    Joint.LEFT_WRIST: "left_palm_skel",
    Joint.LEFT_THUMB_CMC: "left_thproximal_skel",
    Joint.LEFT_THUMB_MCP: "left_thmiddle_skel",
    Joint.LEFT_THUMB_IP: "left_thdistal_skel",
    Joint.LEFT_INDEX_MCP: "left_ffproximal_skel",
    Joint.LEFT_INDEX_PIP: "left_ffmiddle_skel",
    Joint.LEFT_INDEX_DIP: "left_ffdistal_skel",
    Joint.LEFT_MIDDLE_MCP: "left_mfproximal_skel",
    Joint.LEFT_MIDDLE_PIP: "left_mfmiddle_skel",
    Joint.LEFT_MIDDLE_DIP: "left_mfdistal_skel",
    Joint.LEFT_RING_MCP: "left_rfproximal_skel",
    Joint.LEFT_RING_PIP: "left_rfmiddle_skel",
    Joint.LEFT_RING_DIP: "left_rfdistal_skel",
    Joint.LEFT_PINKY_MCP: "left_lfproximal_skel",
    Joint.LEFT_PINKY_PIP: "left_lfmiddle_skel",
    Joint.LEFT_PINKY_DIP: "left_lfdistal_skel",
}

RIGHT_SHADOW_JOINTS = {
    Joint.RIGHT_WRIST: "right_palm_skel",
    Joint.RIGHT_THUMB_CMC: "right_thproximal_skel",
    Joint.RIGHT_THUMB_MCP: "right_thmiddle_skel",
    Joint.RIGHT_THUMB_IP: "right_thdistal_skel",
    Joint.RIGHT_INDEX_MCP: "right_ffproximal_skel",
    Joint.RIGHT_INDEX_PIP: "right_ffmiddle_skel",
    Joint.RIGHT_INDEX_DIP: "right_ffdistal_skel",
    Joint.RIGHT_MIDDLE_MCP: "right_mfproximal_skel",
    Joint.RIGHT_MIDDLE_PIP: "right_mfmiddle_skel",
    Joint.RIGHT_MIDDLE_DIP: "right_mfdistal_skel",
    Joint.RIGHT_RING_MCP: "right_rfproximal_skel",
    Joint.RIGHT_RING_PIP: "right_rfmiddle_skel",
    Joint.RIGHT_RING_DIP: "right_rfdistal_skel",
    Joint.RIGHT_PINKY_MCP: "right_lfproximal_skel",
    Joint.RIGHT_PINKY_PIP: "right_lfmiddle_skel",
    Joint.RIGHT_PINKY_DIP: "right_lfdistal_skel",
}

_POSES = load_npz("robot_models.shadow")

SHADOW_HAND_PRESETS = {
    "left": _POSES["left"],
    "right": _POSES["right"],
}

__all__ = ["LEFT_SHADOW_JOINTS", "RIGHT_SHADOW_JOINTS", "SHADOW_HAND_PRESETS"]
