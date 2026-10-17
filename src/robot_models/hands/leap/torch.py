"""Torch LeapHand."""

from robot_models._backend import model_for_backend

LeapHand = model_for_backend("leap", "torch", module=__name__)

__all__ = ["LeapHand"]
