"""Torch ManoRobot."""

from robot_models._backend import model_for_backend

ManoRobot = model_for_backend("mano", "torch", module=__name__)

__all__ = ["ManoRobot"]
