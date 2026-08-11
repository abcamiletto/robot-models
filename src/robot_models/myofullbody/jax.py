"""JAX MyoFullBody model."""

from robot_models._backend import model_for_backend
from robot_models.myofullbody._model import MyoFullBody as _MyoFullBody

MyoFullBody = model_for_backend(_MyoFullBody, "jax", module=__name__)

__all__ = ["MyoFullBody"]
