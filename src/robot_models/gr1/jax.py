"""JAX GR1 model."""

from robot_models._backend import model_for_backend
from robot_models.gr1._model import GR1 as _GR1

GR1 = model_for_backend(_GR1, "jax", module=__name__)

__all__ = ["GR1"]
