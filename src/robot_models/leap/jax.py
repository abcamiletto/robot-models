"""JAX LEAP Hand model."""

from robot_models._backend import model_for_backend
from robot_models.leap._model import LeapHand as _LeapHand

LeapHand = model_for_backend(_LeapHand, "jax", module=__name__)

__all__ = ["LeapHand"]
