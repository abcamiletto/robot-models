"""Shared model list for cross-model tests."""

from importlib import import_module

from robot_models._catalog import MODEL_SPECS


def backend_model_class(name: str, backend: str):
    spec = MODEL_SPECS[name]
    return getattr(import_module(f"{spec.module}.{backend}"), spec.class_name)


MODELS = [(name, backend_model_class(name, "numpy"), {}) for name in MODEL_SPECS]
