"""Public model discovery, imports and construction."""

import importlib

import model_cases
import pytest

from robot_models import create_model, list_models

CLASSES = {"barrett": "hands.barrett.BarrettHand", "shadow": "hands.shadow.ShadowHand", "g1": "humanoids.g1.G1"}


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


@pytest.mark.fast
@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("unknown", {}),
        ("barrett", {"side": "left"}),
        ("g1", {"side": "left"}),
        ("shadow", {"side": "left", "kinematics": "openai"}),
    ],
)
def test_invalid_models_fail_loudly(name, kwargs):
    with pytest.raises((ValueError, TypeError)):
        create_model(name, **kwargs)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
@pytest.mark.parametrize(("name", "kwargs"), model_cases.RUNTIME_CASES)
def test_named_import_and_factory_select_backend(backend, name, kwargs):
    if backend != "numpy":
        pytest.importorskip(backend)
    package, class_name = CLASSES[name].rsplit(".", 1)
    model_class = getattr(importlib.import_module(f"robot_models.{package}.{backend}"), class_name)
    model = model_class(**kwargs)
    assert type(create_model(name, runtime=backend, **kwargs)) is model_class
    assert model.name == name
    assert model.runtime.name == backend
    with pytest.raises(TypeError):
        model_class(runtime=backend)
