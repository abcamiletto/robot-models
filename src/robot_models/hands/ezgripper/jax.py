"""Jax EZGripper."""

from robot_models._backend import model_for_backend

EZGripper = model_for_backend("ezgripper", "jax", module=__name__)

__all__ = ["EZGripper"]
