"""Authoritative catalog of public models."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ModelSpec:
    """Package and class name of one public model; the name is also its asset folder."""

    module: str
    class_name: str
    asset_version: int = 1


MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "ability": ModelSpec("robot_models.hands.ability", "AbilityHand"),
        "allegro": ModelSpec("robot_models.hands.allegro", "AllegroHand"),
        "barrett": ModelSpec("robot_models.hands.barrett", "BarrettHand", asset_version=3),
        "brainco": ModelSpec("robot_models.hands.brainco", "BrainCoHand"),
        "ezgripper": ModelSpec("robot_models.hands.ezgripper", "EZGripper"),
        "g1": ModelSpec("robot_models.humanoids.g1", "G1"),
        "gr1": ModelSpec("robot_models.humanoids.gr1", "GR1"),
        "h1": ModelSpec("robot_models.humanoids.h1", "H1"),
        "inspire": ModelSpec("robot_models.hands.inspire", "InspireHand"),
        "leap": ModelSpec("robot_models.hands.leap", "LeapHand"),
        "robotiq-3finger": ModelSpec("robot_models.hands.robotiq", "Robotiq3Finger", asset_version=3),
        "sharpa": ModelSpec("robot_models.hands.sharpa", "SharpaHand"),
        "shadow": ModelSpec("robot_models.hands.shadow", "ShadowHand", asset_version=2),
        "t1": ModelSpec("robot_models.humanoids.t1", "T1"),
        "talos": ModelSpec("robot_models.humanoids.talos", "Talos"),
    }
)


__all__ = ["MODEL_SPECS", "ModelSpec"]
