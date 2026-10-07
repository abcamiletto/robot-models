from robot_models._common.pose_assets import load_npz
from robot_models._constants import Joint

LEFT_ABILITY_JOINTS = {
    Joint.LEFT_WRIST: "left_base_skel",
    Joint.LEFT_THUMB_CMC: "left_thumb_cmc_skel",
    Joint.LEFT_THUMB_MCP: "left_thumb_mcp_skel",
    Joint.LEFT_INDEX_MCP: "left_index_mcp_skel",
    Joint.LEFT_INDEX_PIP: "left_index_pip_skel",
    Joint.LEFT_MIDDLE_MCP: "left_middle_mcp_skel",
    Joint.LEFT_MIDDLE_PIP: "left_middle_pip_skel",
    Joint.LEFT_RING_MCP: "left_ring_mcp_skel",
    Joint.LEFT_RING_PIP: "left_ring_pip_skel",
    Joint.LEFT_PINKY_MCP: "left_pinky_mcp_skel",
    Joint.LEFT_PINKY_PIP: "left_pinky_pip_skel",
}

RIGHT_ABILITY_JOINTS = {
    Joint.RIGHT_WRIST: "right_base_skel",
    Joint.RIGHT_THUMB_CMC: "right_thumb_cmc_skel",
    Joint.RIGHT_THUMB_MCP: "right_thumb_mcp_skel",
    Joint.RIGHT_INDEX_MCP: "right_index_mcp_skel",
    Joint.RIGHT_INDEX_PIP: "right_index_pip_skel",
    Joint.RIGHT_MIDDLE_MCP: "right_middle_mcp_skel",
    Joint.RIGHT_MIDDLE_PIP: "right_middle_pip_skel",
    Joint.RIGHT_RING_MCP: "right_ring_mcp_skel",
    Joint.RIGHT_RING_PIP: "right_ring_pip_skel",
    Joint.RIGHT_PINKY_MCP: "right_pinky_mcp_skel",
    Joint.RIGHT_PINKY_PIP: "right_pinky_pip_skel",
}

_POSES = load_npz("robot_models.ability")

ABILITY_HAND_PRESETS = {
    "left": _POSES["left"],
    "right": _POSES["right"],
}

__all__ = ["ABILITY_HAND_PRESETS", "LEFT_ABILITY_JOINTS", "RIGHT_ABILITY_JOINTS"]
