"""Authoritative catalog of canonical robots."""

from types import MappingProxyType

from robot_models._spec import ModelSpec
from robot_models.hands import _models as hands
from robot_models.humanoids import _models as humanoids

MODEL_SPECS = MappingProxyType(dict(sorted((hands.MODELS | humanoids.MODELS).items())))


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def model_spec(name: str) -> ModelSpec:
    """Resolve a canonical name without importing a runtime or loading assets."""
    try:
        return MODEL_SPECS[normalize_name(name)]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}. Available models: {', '.join(MODEL_SPECS)}") from exc


def package(name: str) -> str:
    """The package of a canonical robot's import paths and presets."""
    return f"robot_models.{model_spec(name).family}.{normalize_name(name).replace('-', '_')}"
