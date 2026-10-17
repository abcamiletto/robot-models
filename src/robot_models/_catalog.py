"""Authoritative catalog of public models."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ModelSpec:
    """Public model class and asset folder; unpublished sources require local preparation."""

    module: str
    class_name: str
    downloadable: bool = True


MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "ability": ModelSpec("robot_models.ability", "AbilityHand"),
        "allegro": ModelSpec("robot_models.allegro", "AllegroHand"),
        "barrett-gendexgrasp": ModelSpec("robot_models.baselines", "BarrettGenDexGrasp", downloadable=False),
        "brainco": ModelSpec("robot_models.brainco", "BrainCoHand"),
        "ezgripper-gendexgrasp": ModelSpec("robot_models.baselines", "EZGripperGenDexGrasp", downloadable=False),
        "g1": ModelSpec("robot_models.g1", "G1"),
        "gr1": ModelSpec("robot_models.gr1", "GR1"),
        "h1": ModelSpec("robot_models.h1", "H1"),
        "inspire": ModelSpec("robot_models.inspire", "InspireHand"),
        "leap": ModelSpec("robot_models.leap", "LeapHand"),
        "robotiq-3finger-gendexgrasp": ModelSpec(
            "robot_models.baselines", "Robotiq3FingerGenDexGrasp", downloadable=False
        ),
        "shadow": ModelSpec("robot_models.shadow", "ShadowHand"),
        "shadow-dexgraspnet": ModelSpec("robot_models.baselines", "ShadowDexGraspNet"),
        "shadow-gendexgrasp": ModelSpec("robot_models.baselines", "ShadowGenDexGrasp", downloadable=False),
        "sharpa": ModelSpec("robot_models.sharpa", "SharpaHand"),
        "t1": ModelSpec("robot_models.t1", "T1"),
        "talos": ModelSpec("robot_models.talos", "Talos"),
    }
)


__all__ = ["MODEL_SPECS", "ModelSpec"]
