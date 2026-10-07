"""Snapshot of the stable public API surface."""

from importlib import import_module

import model_cases
import pytest

import robot_models
from robot_models import _catalog as catalog

EXPECTED_ROOT_EXPORTS = (
    "ArrayRuntime",
    "Joint",
    "ParameterRole",
    "ParameterSpec",
    "RigidBodyModel",
    "RotationType",
    "RuntimeName",
    "create_model",
    "list_models",
)

_ROOT_TRANSFORM = {
    "global_rotation": ((3,), "transform", "axis_angle"),
    "global_translation": ((3,), "transform", None),
}

EXPECTED_PARAMETER_SPECS = {
    "ability": {"hand_pose": ((6,), "pose", None), **_ROOT_TRANSFORM},
    "brainco": {"hand_pose": ((6,), "pose", None), **_ROOT_TRANSFORM},
    "g1": {"body_pose": ((29,), "pose", None), **_ROOT_TRANSFORM},
}


@pytest.mark.fast
def test_root_exports_match_snapshot() -> None:
    assert sorted(robot_models.__all__) == sorted(set(robot_models.__all__))
    assert sorted(robot_models.__all__) == list(EXPECTED_ROOT_EXPORTS)
    for name in robot_models.__all__:
        assert getattr(robot_models, name) is not None


@pytest.mark.fast
@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_backend_modules_export_model_class(backend) -> None:
    if backend != "numpy":
        pytest.importorskip(backend)
    for spec in catalog.MODEL_SPECS.values():
        module = import_module(f"{spec.module}.{backend}")
        assert spec.class_name in module.__all__


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_parameter_spec_matches_snapshot(name, model_class, kwargs) -> None:
    model = model_class(**kwargs)
    actual = {key: (tuple(spec.dims), spec.role, spec.rotation_type) for key, spec in model.parameter_spec.items()}
    expected = EXPECTED_PARAMETER_SPECS[name]

    assert list(actual) == list(expected)
    assert actual == expected
