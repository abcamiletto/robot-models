"""Jax BarrettHand."""

from robot_models._backend import model_for_backend

BarrettHand = model_for_backend("barrett", "jax", module=__name__)

__all__ = ["BarrettHand"]
