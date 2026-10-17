"""Shared model list for cross-model tests."""

from importlib import import_module

from robot_models._catalog import MODEL_SPECS

SHADOW_SOURCES = {"shadow-gendexgrasp": "gendexgrasp", "shadow-dexgraspnet": "dexgraspnet"}


def model_selection(name: str):
    """Fixture labels can identify sources; public model names identify families."""
    return ("shadow", {"source": SHADOW_SOURCES[name]}) if name in SHADOW_SOURCES else (name, {})


def backend_model_class(name: str, backend: str):
    family, _ = model_selection(name)
    spec = MODEL_SPECS[family]
    return getattr(import_module(f"{spec.module}.{backend}"), spec.class_name)


MODELS = [
    (name, backend_model_class(name, "numpy"), model_selection(name)[1]) for name in (*MODEL_SPECS, *SHADOW_SOURCES)
]
