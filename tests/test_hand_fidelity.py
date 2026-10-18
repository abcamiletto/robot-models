"""Canonical hands against independent upstream MuJoCo reference poses and surfaces."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from robot_models import create_model

HANDS = ("barrett", "robotiq-3finger", "ezgripper")
REFERENCES = Path(__file__).parent / "reference_hands"


@pytest.mark.parametrize("name", HANDS)
def test_canonical_hands_match_upstream_reference(name):
    """Reference FK imports the upstream URDF/MJCF in MuJoCo, without robot-models."""
    hand = create_model(name)
    directory = hand.xml_path.parent
    manifest = json.loads((directory / "source.json").read_text())
    assert manifest["robot"] == name
    for group in ("source_sha256", "license_sha256"):
        for relative, expected_hash in manifest[group].items():
            assert hashlib.sha256((directory / relative).read_bytes()).hexdigest() == expected_hash
    with np.load(REFERENCES / f"{name}.npz") as reference:
        assert hand.body_names == reference["joint_names"].tolist()
        assert hand.dof_names == reference["pose_names"].tolist()
        np.testing.assert_allclose(hand.dof_limits, reference["limits"], atol=1e-7)
        poses = reference["poses"]
        np.testing.assert_allclose(hand.forward_skeleton(poses), reference["transforms"], atol=2e-6)
        for geometry in ("visual", "collision"):
            prefix = "" if geometry == "visual" else "collision_"
            vertices = np.concatenate([mesh.vertices for mesh in getattr(hand, geometry).meshes])
            triangles = vertices[reference[f"{prefix}point_triangles"]]
            local = np.einsum("pi,pij->pj", reference[f"{prefix}point_weights"], triangles)
            links = np.asarray(getattr(hand, geometry).body_indices)[reference[f"{prefix}point_links"]]
            frames = hand.forward_skeleton(poses)[:, links]
            actual = np.einsum("bpij,pj->bpi", frames[..., :3, :3], local) + frames[..., :3, 3]
            np.testing.assert_allclose(actual, reference[f"{prefix}points"], atol=2e-6)
