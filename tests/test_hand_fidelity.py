"""Canonical hands against independent upstream MuJoCo reference poses and surfaces."""

import hashlib
import inspect
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from robot_models import _assets as assets
from robot_models import create_model, list_models
from robot_models._base import Hand
from robot_models._catalog import MODEL_SPECS

HANDS = ("barrett", "robotiq-3finger", "ezgripper")
REFERENCES = Path(__file__).parent / "reference_hands"


@pytest.mark.fast
def test_one_canonical_model_per_hand():
    names = list_models()
    assert all(name in names for name in HANDS)
    assert not any(name.endswith(("-gendexgrasp", "-dexgraspnet")) for name in names)
    for name in (*HANDS, "shadow"):
        spec = MODEL_SPECS[name]
        cls = getattr(import_module(f"{spec.module}.numpy"), spec.class_name)
        assert issubclass(cls, Hand)
        assert "source" not in inspect.signature(cls).parameters


@pytest.mark.parametrize("name", HANDS)
@pytest.mark.parametrize("runtime", ("numpy", "torch", "jax"))
def test_canonical_hands_match_upstream_reference(name, runtime):
    """Reference FK imports the upstream URDF/MJCF in MuJoCo, without robot-models."""
    hand = create_model(name, runtime=runtime)
    directory = assets.model_dir(name)
    manifest = json.loads((directory / "source.json").read_text())
    assert manifest["robot"] == name
    for group in ("source_sha256", "license_sha256"):
        for relative, expected_hash in manifest[group].items():
            assert hashlib.sha256((directory / relative).read_bytes()).hexdigest() == expected_hash
    xp = hand.runtime
    with np.load(REFERENCES / f"{name}.npz") as reference:
        assert hand.joint_names == reference["joint_names"].tolist()
        assert hand.actuated_joint_names == reference["pose_names"].tolist()
        np.testing.assert_allclose(xp.to_numpy(hand.actuated_joint_limits), reference["limits"], atol=1e-7)
        poses = xp.asarray(reference["poses"], like=hand._assets.vertices)
        np.testing.assert_allclose(xp.to_numpy(hand.forward_skeleton(poses)), reference["transforms"], atol=2e-6)
        triangles = xp.to_numpy(hand._assets.vertices)[reference["point_triangles"]]
        local = np.einsum("pi,pij->pj", reference["point_weights"], triangles)
        frames = xp.to_numpy(hand.forward_links(hand_pose=poses))[:, reference["point_links"]]
        actual = np.einsum("bpij,pj->bpi", frames[..., :3, :3], local) + frames[..., :3, 3]
        np.testing.assert_allclose(actual, reference["points"], atol=2e-6)
