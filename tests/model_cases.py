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


def xml_path(name: str, kwargs: dict):
    """The MJCF file a model of these constructor arguments loads."""
    from robot_models import _assets as assets
    from robot_models._skinned._sources import SOURCES

    spec = MODEL_SPECS[name]
    default = name if spec.body_model is None else SOURCES[spec.body_model].variants[0]
    return assets.model_dir(name) / f"{kwargs.get('side') or kwargs.get('gender') or default}.xml"
