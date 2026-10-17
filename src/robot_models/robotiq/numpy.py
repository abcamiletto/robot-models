"""numpy Robotiq3Finger."""

from robot_models import _backend
from robot_models.robotiq import _model

Robotiq3Finger = _backend.model_for_backend(_model.Robotiq3Finger, "numpy", module=__name__)

__all__ = ["Robotiq3Finger"]
