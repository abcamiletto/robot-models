"""JAX H1 model."""

from robot_models._backend import model_for_backend
from robot_models.h1._model import H1 as _H1

H1 = model_for_backend(_H1, "jax", module=__name__)

__all__ = ["H1"]
