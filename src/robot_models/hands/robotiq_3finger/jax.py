"""Jax Robotiq3Finger."""

from robot_models._backend import model_for_backend

Robotiq3Finger = model_for_backend("robotiq-3finger", "jax", module=__name__)

__all__ = ["Robotiq3Finger"]
