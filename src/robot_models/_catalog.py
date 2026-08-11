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
    """Lazy downloader and output contract for one model family."""

    module: str
    function: str
    output_key: str | None = None


def _model(module: str, class_name: str, **defaults: Any) -> ModelSpec:
    return ModelSpec(module, class_name, MappingProxyType(defaults))


MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "brainco": _model("robot_models.brainco", "BrainCoHand"),
        "g1": _model("robot_models.g1", "G1"),
        "myofullbody": _model("robot_models.myofullbody", "MyoFullBody"),
        "smpl-humanoid": _model("robot_models.smpl_humanoid", "SmplHumanoid"),
    }
)


def _assets(module: str, *names: str) -> dict[str, AssetSpec]:
    return {name: AssetSpec(module) for name in names}


ASSET_SPECS: Mapping[str, AssetSpec] = MappingProxyType(
    {
        **_assets(
            "robot_models.smpl_humanoid._io",
            "smpl-humanoid-humenv",
            "smpl-humanoid-phc",
            "smpl-humanoid-smplsim",
        ),
        **_assets("robot_models.brainco._io", "brainco"),
        **_assets("robot_models.g1._io", "g1"),
        **_assets("robot_models.myofullbody._io", "myofullbody"),
    }
)


DOWNLOAD_SPECS: Mapping[str, DownloadSpec] = MappingProxyType(
    {
        "smpl-humanoid": DownloadSpec("robot_models.smpl_humanoid._io", "download_assets"),
        "brainco": DownloadSpec("robot_models.brainco._io", "download_model", output_key="brainco"),
        "g1": DownloadSpec("robot_models.g1._io", "download_model", output_key="g1"),
        "myofullbody": DownloadSpec(
            "robot_models.myofullbody._io",
            "download_model",
            output_key="myofullbody",
        ),
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
