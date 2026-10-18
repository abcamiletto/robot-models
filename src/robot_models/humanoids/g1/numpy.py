"""Numpy G1."""

from robot_models._backend import model_for_backend

G1 = model_for_backend("g1", "numpy", module=__name__)

__all__ = ["G1"]
