"""JAX Allegro model."""

from robot_models._backend import model_for_backend
from robot_models.allegro._model import AllegroHand as _AllegroHand

AllegroHand = model_for_backend(_AllegroHand, "jax", module=__name__)

__all__ = ["AllegroHand"]
