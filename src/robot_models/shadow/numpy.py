"""NumPy Shadow model."""

from robot_models._backend import model_for_backend
from robot_models.shadow._model import ShadowHand as _ShadowHand

ShadowHand = model_for_backend(_ShadowHand, "numpy", module=__name__)

__all__ = ["ShadowHand"]
