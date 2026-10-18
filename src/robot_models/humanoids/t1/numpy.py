"""Numpy T1."""

from robot_models._backend import model_for_backend

T1 = model_for_backend("t1", "numpy", module=__name__)

__all__ = ["T1"]
