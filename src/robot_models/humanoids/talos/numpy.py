"""Numpy Talos."""

from robot_models._backend import model_for_backend

Talos = model_for_backend("talos", "numpy", module=__name__)

__all__ = ["Talos"]
