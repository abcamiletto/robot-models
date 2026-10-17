"""Numpy AllegroHand."""

from robot_models._backend import model_for_backend

AllegroHand = model_for_backend("allegro", "numpy", module=__name__)

__all__ = ["AllegroHand"]
