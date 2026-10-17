"""Torch ShadowHand."""

from robot_models._backend import model_for_backend

ShadowHand = model_for_backend("shadow", "torch", module=__name__)

__all__ = ["ShadowHand"]
