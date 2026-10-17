"""jax EZGripper."""

from robot_models import _backend
from robot_models.hands.ezgripper import _model

EZGripper = _backend.model_for_backend(_model.EZGripper, "jax", module=__name__)

__all__ = ["EZGripper"]
