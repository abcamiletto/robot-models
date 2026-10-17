"""Jax ShadowHand."""

from robot_models._backend import model_for_backend

ShadowHand = model_for_backend("shadow", "jax", module=__name__)

__all__ = ["ShadowHand"]
