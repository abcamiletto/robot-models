"""Public model discovery, imports and construction."""

import importlib

import model_cases
import pytest

from robot_models import _catalog as catalog
from robot_models import create_model, list_models


@pytest.mark.fast
def test_family_catalog_and_filtering():
    hands = set(list_models(family="hands"))
    humanoids = set(list_models(family="humanoids"))
    assert "shadow" in hands and "g1" in humanoids
    assert hands.isdisjoint(humanoids)
    assert hands | humanoids == set(list_models())
    assert list_models(family="hands", pattern="shadow") == ["shadow"]
    with pytest.raises(ValueError, match="family"):
        list_models(family="unknown")


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.RUNTIME_MODELS)
def test_named_import_and_factory_select_backend(backend, name, model_class, kwargs):
    if backend != "numpy":
        pytest.importorskip(backend)
    module = importlib.import_module(f"{catalog.model_module(name)}.{backend}")
    model_class = getattr(module, catalog.MODEL_SPECS[name].class_name)
    model = model_class(**kwargs)
    assert type(create_model(name, runtime=backend, **kwargs)) is model_class
    assert model.name == name
    assert model.runtime.name == backend
    with pytest.raises(TypeError):
        model_class(runtime=backend)
