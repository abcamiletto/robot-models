"""Shared model list for cross-model tests."""

from importlib import import_module

from robot_models.ability.numpy import AbilityHand
from robot_models.allegro.numpy import AllegroHand
from robot_models.brainco.numpy import BrainCoHand
from robot_models.g1.numpy import G1
from robot_models.gr1.numpy import GR1
from robot_models.h1.numpy import H1
from robot_models.inspire.numpy import InspireHand
from robot_models.leap.numpy import LeapHand
from robot_models.shadow.numpy import ShadowHand
from robot_models.t1.numpy import T1
from robot_models.talos.numpy import Talos

MODELS = [
    ("ability", AbilityHand, {}),
    ("allegro", AllegroHand, {}),
    ("brainco", BrainCoHand, {}),
    ("g1", G1, {}),
    ("gr1", GR1, {}),
    ("h1", H1, {}),
    ("inspire", InspireHand, {}),
    ("leap", LeapHand, {}),
    ("shadow", ShadowHand, {}),
    ("t1", T1, {}),
    ("talos", Talos, {}),
]


def backend_model_class(name: str, backend: str):
    model_class = next(model_class for model_name, model_class, _ in MODELS if model_name == name)
    module = import_module(f"robot_models.{name}.{backend}")
    return getattr(module, model_class.__name__)
