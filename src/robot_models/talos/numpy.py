"""NumPy TALOS model."""

from robot_models._backend import model_for_backend
from robot_models.talos._model import Talos as _Talos

Talos = model_for_backend(_Talos, "numpy", module=__name__)

__all__ = ["Talos"]
