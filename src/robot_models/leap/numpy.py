"""NumPy LEAP Hand model."""

from robot_models._backend import model_for_backend
from robot_models.leap._model import LeapHand as _LeapHand

LeapHand = model_for_backend(_LeapHand, "numpy", module=__name__)

__all__ = ["LeapHand"]
