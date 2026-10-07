"""NumPy Inspire model."""

from robot_models._backend import model_for_backend
from robot_models.inspire._model import InspireHand as _InspireHand

InspireHand = model_for_backend(_InspireHand, "numpy", module=__name__)

__all__ = ["InspireHand"]
