"""Shared model list for cross-model tests."""

from robot_models._catalog import MODEL_SPECS
from robot_models._registry import _model_class


def backend_model_class(name: str, backend: str):
    return _model_class(name, backend)


MODELS = [(name, backend_model_class(name, "numpy"), {}) for name in MODEL_SPECS]

# Shared runtime behavior: a coupled hand, an independent hand and a humanoid.
RUNTIME_MODELS = [
    (name, backend_model_class(name, "numpy"), kwargs)
    for name, kwargs in (("barrett", {}), ("shadow", {"side": "left"}), ("g1", {}))
]
