"""Authoritative catalog of public models."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ModelSpec:
    """Package and class name of one public model; the name is also its asset folder."""

    module: str
    class_name: str


MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "ability": ModelSpec("robot_models.ability", "AbilityHand"),
        "allegro": ModelSpec("robot_models.allegro", "AllegroHand"),
        "brainco": ModelSpec("robot_models.brainco", "BrainCoHand"),
        "g1": ModelSpec("robot_models.g1", "G1"),
        "gr1": ModelSpec("robot_models.gr1", "GR1"),
        "h1": ModelSpec("robot_models.h1", "H1"),
        "inspire": ModelSpec("robot_models.inspire", "InspireHand"),
        "leap": ModelSpec("robot_models.leap", "LeapHand"),
        "shadow": ModelSpec("robot_models.shadow", "ShadowHand"),
        "t1": ModelSpec("robot_models.t1", "T1"),
        "talos": ModelSpec("robot_models.talos", "Talos"),
    }
)


__all__ = ["MODEL_SPECS", "ModelSpec"]
