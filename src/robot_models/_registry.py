"""Lazy model construction from the public catalog."""

from __future__ import annotations

from fnmatch import fnmatchcase
from importlib import import_module
from typing import Any

from robot_models import _catalog as catalog
from robot_models._base import RigidBodyModel
from robot_models._runtime import RuntimeName
from robot_models._spec import Family


def create_model(
    model_name: str,
    *,
    runtime: RuntimeName = "numpy",
    **kwargs: Any,
) -> RigidBodyModel:
    """
    Create a model from its public catalog name.

    Args:
        model_name: Name returned by :func:`list_models`. Names are
            case-insensitive, and underscores are treated as hyphens.
        runtime: Array backend name.
        **kwargs: Model-specific constructor options.

    Returns:
        The requested articulated model.

    Raises:
        ValueError: If ``model_name`` is unknown.
    """
    return _model_class(model_name, runtime)(**kwargs)


def _model_class(model_name: str, runtime: RuntimeName) -> type[RigidBodyModel]:
    """Resolve the shared catalog entry without loading its assets."""
    if runtime not in ("numpy", "torch", "jax"):
        raise ValueError(f"Unknown runtime {runtime!r}. Expected numpy, torch, or jax.")
    spec = catalog.model_spec(model_name)
    module = import_module(f"{catalog.model_module(model_name)}.{runtime}")
    return getattr(module, spec.class_name)


def list_models(*, family: Family | None = None, pattern: str | None = None) -> list[str]:
    """
    List public model factory names.

    Args:
        family: Optional ``"hands"`` or ``"humanoids"`` filter.
        pattern: Optional case-insensitive shell-style pattern such as ``"brainco*"``.

    Returns:
        Sorted matching model names.
    """
    if family not in (None, "hands", "humanoids"):
        raise ValueError(f"Unknown model family {family!r}")
    names = [name for name, spec in catalog.MODEL_SPECS.items() if family is None or spec.family == family]
    if pattern is None:
        return names
    pattern = catalog.normalize_name(pattern)
    return [name for name in names if fnmatchcase(name, pattern)]


__all__ = ["create_model", "list_models"]
