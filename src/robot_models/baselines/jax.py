"""jax baseline-native hands."""

from robot_models import _backend
from robot_models.baselines import _model

BarrettGenDexGrasp = _backend.model_for_backend(_model.BarrettGenDexGrasp, "jax", module=__name__)
Robotiq3FingerGenDexGrasp = _backend.model_for_backend(_model.Robotiq3FingerGenDexGrasp, "jax", module=__name__)
EZGripperGenDexGrasp = _backend.model_for_backend(_model.EZGripperGenDexGrasp, "jax", module=__name__)
ShadowGenDexGrasp = _backend.model_for_backend(_model.ShadowGenDexGrasp, "jax", module=__name__)
ShadowDexGraspNet = _backend.model_for_backend(_model.ShadowDexGraspNet, "jax", module=__name__)

__all__ = [
    "BarrettGenDexGrasp",
    "EZGripperGenDexGrasp",
    "Robotiq3FingerGenDexGrasp",
    "ShadowDexGraspNet",
    "ShadowGenDexGrasp",
]
