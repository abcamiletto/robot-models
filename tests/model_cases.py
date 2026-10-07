"""Shared model list for cross-model tests."""

from importlib import import_module

from robot_models.allegro.numpy import AllegroHand
from robot_models.brainco.numpy import BrainCoHand
from robot_models.g1.numpy import G1

MODELS = [
    ("allegro", AllegroHand, {}),
    ("brainco", BrainCoHand, {}),
    ("g1", G1, {}),
]


def backend_model_class(name: str, backend: str):
    model_class = next(model_class for model_name, model_class, _ in MODELS if model_name == name)
    module = import_module(f"robot_models.{name}.{backend}")
    return getattr(module, model_class.__name__)
