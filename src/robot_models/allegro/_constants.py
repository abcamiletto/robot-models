from robot_models._common.pose_assets import load_npz
from robot_models._constants import Joint

LEFT_ALLEGRO_JOINTS = {
    Joint.LEFT_WRIST: "left_palm_skel",
    Joint.LEFT_THUMB_CMC: "left_th_base_skel",
    Joint.LEFT_THUMB_MCP: "left_th_medial_skel",
    Joint.LEFT_THUMB_IP: "left_th_distal_skel",
    Joint.LEFT_INDEX_MCP: "left_ff_proximal_skel",
    Joint.LEFT_INDEX_PIP: "left_ff_medial_skel",
    Joint.LEFT_INDEX_DIP: "left_ff_distal_skel",
    Joint.LEFT_MIDDLE_MCP: "left_mf_proximal_skel",
    Joint.LEFT_MIDDLE_PIP: "left_mf_medial_skel",
    Joint.LEFT_MIDDLE_DIP: "left_mf_distal_skel",
    Joint.LEFT_RING_MCP: "left_rf_proximal_skel",
    Joint.LEFT_RING_PIP: "left_rf_medial_skel",
    Joint.LEFT_RING_DIP: "left_rf_distal_skel",
}

RIGHT_ALLEGRO_JOINTS = {
    Joint.RIGHT_WRIST: "right_palm_skel",
    Joint.RIGHT_THUMB_CMC: "right_th_base_skel",
    Joint.RIGHT_THUMB_MCP: "right_th_medial_skel",
    Joint.RIGHT_THUMB_IP: "right_th_distal_skel",
    Joint.RIGHT_INDEX_MCP: "right_ff_proximal_skel",
    Joint.RIGHT_INDEX_PIP: "right_ff_medial_skel",
    Joint.RIGHT_INDEX_DIP: "right_ff_distal_skel",
    Joint.RIGHT_MIDDLE_MCP: "right_mf_proximal_skel",
    Joint.RIGHT_MIDDLE_PIP: "right_mf_medial_skel",
    Joint.RIGHT_MIDDLE_DIP: "right_mf_distal_skel",
    Joint.RIGHT_RING_MCP: "right_rf_proximal_skel",
    Joint.RIGHT_RING_PIP: "right_rf_medial_skel",
    Joint.RIGHT_RING_DIP: "right_rf_distal_skel",
}

_POSES = load_npz("robot_models.allegro")

ALLEGRO_HAND_PRESETS = {
    "left": _POSES["left"],
    "right": _POSES["right"],
}

__all__ = ["ALLEGRO_HAND_PRESETS", "LEFT_ALLEGRO_JOINTS", "RIGHT_ALLEGRO_JOINTS"]
