"""Torch SharpaHand."""

from robot_models._backend import model_for_backend

SharpaHand = model_for_backend("sharpa", "torch", module=__name__)

__all__ = ["SharpaHand"]
