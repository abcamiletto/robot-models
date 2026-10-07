"""NumPy T1 model."""

from robot_models._backend import model_for_backend
from robot_models.t1._model import T1 as _T1

T1 = model_for_backend(_T1, "numpy", module=__name__)

__all__ = ["T1"]
