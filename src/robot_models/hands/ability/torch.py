"""Torch AbilityHand."""

from robot_models._backend import model_for_backend

AbilityHand = model_for_backend("ability", "torch", module=__name__)

__all__ = ["AbilityHand"]
