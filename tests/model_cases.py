"""Models, constructor arguments and conventions shared by cross-model tests."""

import numpy as np

from robot_models import list_models
from robot_models._catalog import MODEL_SPECS  # only to enumerate each model's sides or genders

# MuJoCo's Z-up axes expressed in the Y-up model frame.
M = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

# Shared runtime behavior: a coupled hand, an independent hand and a humanoid.
RUNTIME_CASES = [("barrett", {}), ("shadow", {"side": "left"}), ("g1", {})]


def variants(name: str) -> list[dict]:
    """The constructor arguments of each MJCF file a model offers, its other kinematics included."""
    spec = MODEL_SPECS[name]
    key = "side" if spec.family == "hands" else "gender"
    own = [{key: variant} for variant in spec.variants] or [{}]
    return own + [{key: v, "kinematics": k} for k, sides in spec.kinematics.items() for v in sides]


CASES = [(name, kwargs) for name in list_models() for kwargs in variants(name)]
HAND_CASES = [
    (name, kwargs["side"])
    for name, kwargs in CASES
    if name in list_models(family="hands") and "kinematics" not in kwargs
]


def pose_name(model) -> str:
    """The pose parameter: a hand's ``hand_pose`` or a humanoid's ``body_pose``."""
    return "hand_pose" if "hand_pose" in model.parameter_spec else "body_pose"
