"""Torch Talos."""

from robot_models._backend import model_for_backend

Talos = model_for_backend("talos", "torch", module=__name__)

__all__ = ["Talos"]
