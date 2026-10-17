"""Numpy InspireHand."""

from robot_models._backend import model_for_backend

InspireHand = model_for_backend("inspire", "numpy", module=__name__)

__all__ = ["InspireHand"]
