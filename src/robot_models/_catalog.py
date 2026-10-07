"""Authoritative catalog of public models and configurable assets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Lazy import and constructor defaults for one public factory name."""

    module: str
    class_name: str
    defaults: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssetSpec:
    """Validation route for one persistent asset configuration key."""

    validation_module: str


@dataclass(frozen=True)
class DownloadSpec:
    """Lazy downloader that returns the asset path for one model family."""

    module: str
    function: str


def _model(module: str, class_name: str, **defaults: Any) -> ModelSpec:
    return ModelSpec(module, class_name, MappingProxyType(defaults))


MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "brainco": _model("robot_models.brainco", "BrainCoHand"),
        "g1": _model("robot_models.g1", "G1"),
        "leap": _model("robot_models.leap", "LeapHand"),
    }
)


ASSET_SPECS: Mapping[str, AssetSpec] = MappingProxyType(
    {
        "brainco": AssetSpec("robot_models.brainco._io"),
        "g1": AssetSpec("robot_models.g1._io"),
        "leap": AssetSpec("robot_models.leap._io"),
    }
)


DOWNLOAD_SPECS: Mapping[str, DownloadSpec] = MappingProxyType(
    {
        "brainco": DownloadSpec("robot_models.brainco._io", "download_model"),
        "g1": DownloadSpec("robot_models.g1._io", "download_model"),
        "leap": DownloadSpec("robot_models.leap._io", "download_model"),
    }
)


__all__ = [
    "ASSET_SPECS",
    "DOWNLOAD_SPECS",
    "MODEL_SPECS",
    "AssetSpec",
    "DownloadSpec",
    "ModelSpec",
]
