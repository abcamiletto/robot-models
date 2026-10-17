"""Jax AbilityHand."""

from robot_models._backend import model_for_backend

AbilityHand = model_for_backend("ability", "jax", module=__name__)

__all__ = ["AbilityHand"]
