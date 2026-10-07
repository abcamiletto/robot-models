from robot_models._common.pose_assets import load_npz
from robot_models._constants import Joint

LEFT_LEAP_JOINTS = {
    Joint.LEFT_WRIST: "left_palm_skel",
    Joint.LEFT_THUMB_CMC: "left_th_cmc_skel",
    Joint.LEFT_THUMB_MCP: "left_th_mcp_skel",
    Joint.LEFT_THUMB_IP: "left_th_ipl_skel",
    Joint.LEFT_INDEX_MCP: "left_if_mcp_skel",
    Joint.LEFT_INDEX_PIP: "left_if_pip_skel",
    Joint.LEFT_INDEX_DIP: "left_if_dip_skel",
    Joint.LEFT_MIDDLE_MCP: "left_mf_mcp_skel",
    Joint.LEFT_MIDDLE_PIP: "left_mf_pip_skel",
    Joint.LEFT_MIDDLE_DIP: "left_mf_dip_skel",
    Joint.LEFT_RING_MCP: "left_rf_mcp_skel",
    Joint.LEFT_RING_PIP: "left_rf_pip_skel",
    Joint.LEFT_RING_DIP: "left_rf_dip_skel",
}

RIGHT_LEAP_JOINTS = {
    Joint.RIGHT_WRIST: "right_palm_skel",
    Joint.RIGHT_THUMB_CMC: "right_th_cmc_skel",
    Joint.RIGHT_THUMB_MCP: "right_th_mcp_skel",
    Joint.RIGHT_THUMB_IP: "right_th_ipl_skel",
    Joint.RIGHT_INDEX_MCP: "right_if_mcp_skel",
    Joint.RIGHT_INDEX_PIP: "right_if_pip_skel",
    Joint.RIGHT_INDEX_DIP: "right_if_dip_skel",
    Joint.RIGHT_MIDDLE_MCP: "right_mf_mcp_skel",
    Joint.RIGHT_MIDDLE_PIP: "right_mf_pip_skel",
    Joint.RIGHT_MIDDLE_DIP: "right_mf_dip_skel",
    Joint.RIGHT_RING_MCP: "right_rf_mcp_skel",
    Joint.RIGHT_RING_PIP: "right_rf_pip_skel",
    Joint.RIGHT_RING_DIP: "right_rf_dip_skel",
}

_POSES = load_npz("robot_models.leap")

LEAP_HAND_PRESETS = {
    "left": _POSES["left"],
    "right": _POSES["right"],
}

__all__ = ["LEAP_HAND_PRESETS", "LEFT_LEAP_JOINTS", "RIGHT_LEAP_JOINTS"]
