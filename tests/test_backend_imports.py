"""Named backend imports share each family's implementation."""

import importlib
import inspect

import pytest

from robot_models import _catalog as catalog


@pytest.mark.fast
def test_backend_model_signatures() -> None:
    for name, spec in catalog.MODEL_SPECS.items():
        base_module = importlib.import_module(f"robot_models.{spec.family}._base")
        base_class = base_module.Hand if spec.family == "hands" else base_module.Humanoid
        for backend in ("numpy", "torch", "jax"):
            module_name = f"{catalog.model_module(name)}.{backend}"
            model_class = getattr(importlib.import_module(module_name), spec.class_name)
            parameters = inspect.signature(model_class).parameters
            assert issubclass(model_class, base_class)
            assert model_class.forward_skeleton is base_class.forward_skeleton
            assert model_class.__module__ == module_name
            assert "runtime" not in parameters
            assert "name" not in parameters
            with pytest.raises(TypeError, match=rf"{spec.class_name}\.__init__\(\)"):
                model_class(runtime=backend)
