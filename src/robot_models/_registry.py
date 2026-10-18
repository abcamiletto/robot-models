"""Model classes and construction from the public catalog."""

from __future__ import annotations

import functools
from fnmatch import fnmatchcase
from typing import Any

from robot_models import _catalog as catalog
from robot_models._base import RigidBodyModel
from robot_models._runtime import JaxRuntime, NumpyRuntime, RuntimeName, TorchRuntime
from robot_models._skinned._model import SkinnedHand, SkinnedHumanoid
from robot_models._spec import Family
from robot_models.hands._base import Hand
from robot_models.humanoids._base import Humanoid

_RUNTIMES = {"numpy": NumpyRuntime, "torch": TorchRuntime, "jax": JaxRuntime}
_FAMILIES = {("hands", False): Hand, ("hands", True): SkinnedHand}
_FAMILIES |= {("humanoids", False): Humanoid, ("humanoids", True): SkinnedHumanoid}


def model_class(model_name: str, runtime: RuntimeName) -> type[RigidBodyModel]:
    """The class of a catalog model on a runtime: the one ``create_model`` builds and its import path names.

    For example ``model_class("shadow", "torch")`` is ``robot_models.hands.shadow.torch.ShadowHand``.
    """
    return _model_class(catalog.normalize_name(model_name), runtime)


@functools.cache
def _model_class(name: str, runtime: RuntimeName) -> type[RigidBodyModel]:
    spec = catalog.model_spec(name)
    if runtime not in _RUNTIMES:
        raise ValueError(f"Unknown runtime {runtime!r}. Expected numpy, torch, or jax.")
    family = _FAMILIES[spec.family, spec.source is not None]
    bases: tuple[type, ...] = (family,)
    if runtime == "torch":
        from torch import nn

        bases = (family, nn.Module)
    namespace = {
        "__module__": f"{catalog.package(name)}.{runtime}",
        "__qualname__": spec.class_name,
        "__doc__": f"{spec.class_name} on {runtime} arrays.\n\n{family.__doc__}",
        "_name": name,
        "_runtime": _RUNTIMES[runtime](),
    }
    return type(spec.class_name, bases, namespace)


def create_model(model_name: str, *, runtime: RuntimeName = "numpy", **kwargs: Any) -> RigidBodyModel:
    """
    Create a model from its public catalog name.

    Args:
        model_name: Name returned by :func:`list_models`. Names are
            case-insensitive, and underscores are treated as hyphens.
        runtime: Array backend name.
        **kwargs: The model's constructor arguments, such as ``side``.

    Returns:
        The requested articulated model.

    Raises:
        ValueError: If ``model_name`` or ``runtime`` is unknown.
    """
    return model_class(model_name, runtime)(**kwargs)


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


__all__ = ["create_model", "list_models", "model_class"]
