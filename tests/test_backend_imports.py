"""Backend-specific public model imports."""

import importlib
import inspect

import pytest

from robot_models import _catalog as catalog
from robot_models.hands._base import Hand
from robot_models.humanoids._base import Humanoid


@pytest.mark.fast
def test_backend_model_signatures() -> None:
    for spec in catalog.MODEL_SPECS.values():
        package = spec.module
        base_class = getattr(importlib.import_module(f"{package}._model"), spec.class_name)
        family = "hands" if issubclass(base_class, Hand) else "humanoids"
        assert package.startswith(f"robot_models.{family}.")
        assert issubclass(base_class, Hand if family == "hands" else Humanoid)
        for backend in ("numpy", "torch", "jax"):
            model_class = getattr(importlib.import_module(f"{package}.{backend}"), spec.class_name)
            parameters = inspect.signature(model_class).parameters

            assert issubclass(model_class, base_class)
            assert model_class.__module__ == f"{package}.{backend}"
            assert "runtime" not in parameters
            assert "kernel_backend" not in parameters
