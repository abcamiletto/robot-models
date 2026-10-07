"""PyTorch Ability Hand model."""

from robot_models._backend import model_for_backend
from robot_models.ability._model import AbilityHand as _AbilityHand

AbilityHand = model_for_backend(_AbilityHand, "torch", module=__name__)

__all__ = ["AbilityHand"]
