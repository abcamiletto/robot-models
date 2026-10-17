"""jax BarrettHand."""

from robot_models import _backend
from robot_models.hands.barrett import _model

BarrettHand = _backend.model_for_backend(_model.BarrettHand, "jax", module=__name__)

__all__ = ["BarrettHand"]
