"""Jax GR1."""

from robot_models._backend import model_for_backend

GR1 = model_for_backend("gr1", "jax", module=__name__)

__all__ = ["GR1"]
