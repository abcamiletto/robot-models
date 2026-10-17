"""Jax H1."""

from robot_models._backend import model_for_backend

H1 = model_for_backend("h1", "jax", module=__name__)

__all__ = ["H1"]
