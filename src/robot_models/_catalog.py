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


@dataclass(frozen=True)
class AssetSpec:
    """One source's hub folder, cache/config key and publication status."""

    name: str
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
        "sharpa": ModelSpec("robot_models.sharpa", "SharpaHand"),
        "t1": ModelSpec("robot_models.t1", "T1"),
        "talos": ModelSpec("robot_models.talos", "Talos"),
    }
)


MODEL_SOURCES: Mapping[str, Mapping[str, AssetSpec]] = MappingProxyType(
    {
        "shadow": MappingProxyType(
            {
                "menagerie": AssetSpec("shadow"),
                "gendexgrasp": AssetSpec("shadow-gendexgrasp", downloadable=False),
                "dexgraspnet": AssetSpec("shadow-dexgraspnet", downloadable=False),
            }
        )
    }
)


def asset_spec(model: str, source: str | None = None) -> AssetSpec:
    """Resolve a model/source selection without treating source folders as models."""
    if model not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model!r}")
    if source is None:
        return AssetSpec(model, MODEL_SPECS[model].downloadable)
    sources = MODEL_SOURCES.get(model, {})
    if source not in sources:
        raise ValueError(f"Unknown source {source!r} for {model!r}. Available sources: {tuple(sources)}")
    return sources[source]


__all__ = ["MODEL_SOURCES", "MODEL_SPECS", "AssetSpec", "ModelSpec", "asset_spec"]
