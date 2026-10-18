"""Jax BrainCoHand."""

from robot_models._backend import model_for_backend

BrainCoHand = model_for_backend("brainco", "jax", module=__name__)

__all__ = ["BrainCoHand"]
