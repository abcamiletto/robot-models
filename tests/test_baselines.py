"""Source conversion regressions and independent upstream numerical fixtures.

Unlicensed source assets/derived geometry fixtures stay local. The CLI audit
fails if any upstream comparison is unavailable; these optional fixture tests
also run in every backend once local bundles have been prepared.
"""

import importlib.util
import json
import os
from pathlib import Path

import model_cases
import numpy as np
import pytest
import trimesh

from robot_models import _assets as assets
from robot_models import _config as config
from robot_models import create_model
from robot_models._catalog import asset_spec
from robot_models._common.mjcf import MUJOCO_TO_MODEL as C
from robot_models.baselines import source_assets
from robot_models.baselines._source import BASELINE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_baseline_assets", REPO_ROOT / "scripts/prepare_baseline_assets.py"
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)
RUNTIME_NAMES = tuple(name for name in BASELINE_NAMES if name != "allegro-gendexgrasp")


@pytest.mark.fast
def test_source_conversion_preserves_rest_axes_and_independent_mimics(tmp_path):
    """Nonidentity wrist rotation and negative axes must survive; source mimic stays independent."""
    mesh_dir = tmp_path / "original_meshes"
    mesh_dir.mkdir()
    mesh = trimesh.creation.box(extents=(0.01, 0.02, 0.03))
    mesh.apply_translation((0.003, 0.004, 0.005))
    mesh.export(mesh_dir / "test.obj")
    xml = """<robot name="fixture">
      <link name="forearm"/>
      <link name="palm"><visual><origin xyz=".01 .02 .03" rpy="0 0 .3"/>
        <geometry><mesh filename="package://fixture/test.obj" scale="2 3 4"/></geometry>
      </visual></link>
      <link name="tip"/>
      <joint name="wrist" type="revolute"><parent link="forearm"/><child link="palm"/>
        <origin xyz=".02 -.01 .03" rpy="0 0 1.5707963267948966"/><axis xyz="0 -1 0"/>
        <limit lower="-.5" upper="1"/></joint>
      <joint name="distal" type="revolute"><parent link="palm"/><child link="tip"/>
        <origin xyz=".01 .02 .05" rpy=".2 .1 0"/><axis xyz="-1 0 0"/>
        <limit lower="-.4" upper=".9"/><mimic joint="wrist" multiplier=".5"/></joint>
      </robot>"""
    description = tmp_path / "original.urdf"
    description.write_text(xml)
    output = tmp_path / "runtime"
    output.mkdir()
    mapping = prepare.convert_urdf("shadowhand", description, mesh_dir, output)
    (output / "source.json").write_text(json.dumps({"upstream": prepare.GENDEX_URL, "robot": "shadowhand"}))
    assert description.read_text() == xml
    assert mapping["source_mimic"] == {"distal": {"joint": "wrist", "multiplier": ".5"}}
    model = create_model("shadow", source="gendexgrasp", model_path=output)
    assert model.actuated_joint_names == ["wrist", "distal"]
    poses = np.array([[0.0, 0.0], [0.21, -0.13]], dtype=np.float32)
    actual = model.forward_skeleton(poses)
    basis = np.eye(4)
    basis[:3, :3] = C
    wrist_rest = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    wrist_rest[:3, 3] = [0.02, -0.01, 0.03]
    tip_rest = trimesh.transformations.euler_matrix(0.2, 0.1, 0, axes="sxyz")
    tip_rest[:3, 3] = [0.01, 0.02, 0.05]
    for row, (q0, q1) in enumerate(poses):
        palm = wrist_rest @ trimesh.transformations.rotation_matrix(q0, [0, -1, 0])
        tip = palm @ tip_rest @ trimesh.transformations.rotation_matrix(q1, [-1, 0, 0])
        np.testing.assert_allclose(actual[row, 1], basis @ palm @ basis.T, atol=1e-7)
        np.testing.assert_allclose(actual[row, 2], basis @ tip @ basis.T, atol=1e-7)
    # GenDexGrasp's Shadow Rx(pi/2) is applied after scale and before visual origin.
    local = mesh.vertices * [2, 3, 4]
    local = local @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
    rotation = trimesh.transformations.rotation_matrix(0.3, [0, 0, 1])[:3, :3]
    expected = (local @ rotation.T + [0.01, 0.02, 0.03]) @ C.T
    actual = np.asarray(model.link_meshes[0].vertices)
    np.testing.assert_allclose(actual[np.lexsort(actual.T)], expected[np.lexsort(expected.T)], atol=1e-7)


@pytest.mark.fast
def test_dexgraspnet_conversion_retains_annotation_frames_and_bevelled_boxes(tmp_path):
    (tmp_path / "meshes").mkdir()
    # Asymmetric fixture makes an ideal-box substitution detectable.
    box = trimesh.creation.icosphere(subdivisions=0)
    box.export(tmp_path / "meshes/box.obj")
    description = tmp_path / "source.xml"
    description.write_text("""<mujoco><worldbody><body name="robot0:palm">
      <geom type="box" size=".03 .01 .04" pos=".01 .02 .03"/>
      <body name="robot0:finger" pos=".01 0 .04">
        <joint name="J0" axis="-1 0 0" range="-.2 1.5"/>
        <geom type="capsule" size=".007 .025" pos="0 0 .025"/>
      </body></body></worldbody></mujoco>""")
    output = tmp_path / "runtime"
    output.mkdir()
    prepare.convert_mjcf(description, output)
    (output / "source.json").write_text(json.dumps({"upstream": prepare.DEX_URL}))
    model = create_model("shadow", source="dexgraspnet", model_path=output)
    assert model.joint_names == ["robot0:palm", "robot0:finger", "robot0:finger_child"]
    assert model.link_names == ["robot0:palm", "robot0:finger_child"]
    skeleton = model.forward_skeleton(np.array([0.3], dtype=np.float32))
    np.testing.assert_allclose(skeleton[1, :3, :3], np.eye(3))
    rotation = trimesh.transformations.rotation_matrix(0.3, [-1, 0, 0])[:3, :3]
    np.testing.assert_allclose(skeleton[2, :3, :3], C @ rotation @ C.T, atol=1e-7)
    expected = (box.vertices * [0.03, 0.01, 0.04] + [0.01, 0.02, 0.03]) @ C.T
    actual = np.asarray(model.link_meshes[0].vertices)
    np.testing.assert_allclose(actual[np.lexsort(actual.T)], expected[np.lexsort(expected.T)], atol=1e-7)
    vertices = np.asarray(model.link_meshes[1].vertices) @ C
    nearest = np.zeros_like(vertices)
    nearest[:, 2] = np.clip(vertices[:, 2], 0, 0.05)
    np.testing.assert_allclose(np.linalg.norm(vertices - nearest, axis=1), 0.007, atol=1e-7)


@pytest.mark.fast
def test_source_bundle_paths_hashes_and_native_annotations(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "hand.urdf").write_text('<robot name="original"/>')
    (source / "contacts.json").write_text('{"tip": [1, 2, 3, 4]}')
    manifest = {
        "description": "source/hand.urdf",
        "mesh_dir": "source",
        "annotations": {"contacts": "source/contacts.json"},
    }
    prepare.write_bundle(tmp_path, manifest, "Fixture assets authored for this test.\n")
    bundle = source_assets("barrett-gendexgrasp", model_path=tmp_path)
    assert bundle.root == source
    assert json.loads(bundle.annotations["contacts"].read_text()) == {"tip": [1, 2, 3, 4]}
    bundle.verify()
    notice = tmp_path / "LICENSE.md"
    notice.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        bundle.verify()
    notice.write_text("Fixture assets authored for this test.\n")
    (source / "hand.urdf").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        bundle.verify()
    manifest["description"] = "../outside.urdf"
    (tmp_path / "source.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        source_assets("barrett-gendexgrasp", model_path=tmp_path)


@pytest.mark.fast
@pytest.mark.parametrize(
    "name", ("barrett-gendexgrasp", "robotiq-3finger-gendexgrasp", "ezgripper-gendexgrasp", "shadow-gendexgrasp")
)
def test_unpublished_variants_give_local_preparation_instructions(name, tmp_path):
    family, options = model_cases.model_selection(name)
    with pytest.raises(FileNotFoundError, match="Prepare a local bundle"):
        assets.download(family, tmp_path, **options)


@pytest.mark.parametrize("name", RUNTIME_NAMES)
@pytest.mark.parametrize("runtime", ["numpy", "torch", "jax"])
def test_matches_frozen_upstream_reference(name, runtime):
    directory = Path(os.environ.get("ROBOT_MODELS_BASELINE_FIXTURES", REPO_ROOT / "results/fidelity-fixtures"))
    fixture = directory / f"{name}.npz"
    if not fixture.exists():
        pytest.skip("Generate independent reference fixtures with scripts/check_baseline_fidelity.py")
    family, options = model_cases.model_selection(name)
    source = options.get("source")
    spec = asset_spec(family, source)
    if config.get_model_path(family, source=source) is None and not (assets.get_cache_dir() / spec.name).is_dir():
        pytest.skip(f"Prepare local {name} source assets; see docs/models/baselines.md")
    model = create_model(family, runtime=runtime, **options)
    with np.load(fixture) as reference:
        assert model.joint_names == reference["joint_names"].tolist()
        assert model.actuated_joint_names == reference["pose_names"].tolist()
        np.testing.assert_allclose(model.runtime.to_numpy(model.actuated_joint_limits), reference["limits"], atol=1e-7)
        poses = model.runtime.asarray(reference["poses"], like=model._assets.vertices)
        skeleton = model.runtime.to_numpy(model.forward_skeleton(poses))
        np.testing.assert_allclose(skeleton, reference["transforms"], atol=2e-6, rtol=2e-6)
        if "point_indices" in reference:
            # Points inside independently selected source triangles, through articulated FK.
            local = np.einsum(
                "pi,pij->pj",
                reference["point_weights"],
                model.runtime.to_numpy(model._assets.vertices)[reference["point_indices"]],
            )
            frames = skeleton[:, reference["point_bodies"]]
            points = np.einsum("bpij,pj->bpi", frames[:, :, :3, :3], local) + frames[:, :, :3, 3]
            np.testing.assert_allclose(points, reference["point_expected"], atol=2e-6)
        for key in ("contacts", "penetration", "contact_basis"):
            if f"{key}_local" not in reference:
                continue
            frames = skeleton[:, reference[f"{key}_bodies"]]
            points = np.einsum("bpij,pj->bpi", frames[:, :, :3, :3], reference[f"{key}_local"]) + frames[:, :, :3, 3]
            np.testing.assert_allclose(points, reference[f"{key}_expected"], atol=2e-6)

        if "capsule_names" in reference:
            for body, radius, height in zip(
                reference["capsule_names"], reference["capsule_radii"], reference["capsule_heights"], strict=True
            ):
                vertices = np.asarray(model.link_meshes[model.link_names.index(str(body))].vertices) @ C
                nearest = np.zeros_like(vertices)
                nearest[:, 2] = np.clip(vertices[:, 2], 0, height)
                np.testing.assert_allclose(np.linalg.norm(vertices - nearest, axis=1), radius, atol=2e-7)
