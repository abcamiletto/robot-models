"""Torch SmplRobot."""

from robot_models._backend import model_for_backend

SmplRobot = model_for_backend("smpl", "torch", module=__name__)

__all__ = ["SmplRobot"]
