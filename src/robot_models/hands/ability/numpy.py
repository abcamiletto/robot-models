"""Numpy AbilityHand."""

from robot_models._backend import model_for_backend

AbilityHand = model_for_backend("ability", "numpy", module=__name__)

__all__ = ["AbilityHand"]
