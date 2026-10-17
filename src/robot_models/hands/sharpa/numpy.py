"""Numpy SharpaHand."""

from robot_models._backend import model_for_backend

SharpaHand = model_for_backend("sharpa", "numpy", module=__name__)

__all__ = ["SharpaHand"]
