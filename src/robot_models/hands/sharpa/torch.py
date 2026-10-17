"""PyTorch Sharpa Wave hand."""

from robot_models._backend import model_for_backend
from robot_models.hands.sharpa._model import SharpaHand as _SharpaHand

SharpaHand = model_for_backend(_SharpaHand, "torch", module=__name__)

__all__ = ["SharpaHand"]
