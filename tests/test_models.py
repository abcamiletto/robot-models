"""Every model against MuJoCo's own reading of its MJCF file."""

import model_cases
import numpy as np
import pytest

from robot_models import _assets as assets
from robot_models._base import Hand
from robot_models._common.mjcf import MUJOCO_TO_MODEL

CASES = [
    (name, kwargs)
    for name, model_class, _ in model_cases.MODELS
    for kwargs in ([{"side": "right"}, {"side": "left"}] if issubclass(model_class, Hand) else [{}])
]


@pytest.fixture(params=CASES, ids=lambda case: "-".join([case[0], *case[1].values()]))
def case(request):
    """A model, MuJoCo's model of the same file, and both posed with one random pose."""
    mujoco = pytest.importorskip("mujoco")
    name, kwargs = request.param
    model = model_cases.backend_model_class(name, "numpy")(**kwargs)
    xml_path = assets.model_dir(name) / f"{kwargs.get('side', name)}.xml"
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(mj_model)

    pose = np.random.default_rng(0).uniform(-0.6, 0.6, model.num_dofs)
    pose = np.clip(pose, *np.asarray(model.actuated_joint_limits).T).astype(np.float32)
    hinge_angles = model.to_qpos(pose)[7:]
    data.qpos[mj_model.nq - len(hinge_angles) :] = hinge_angles
    mujoco.mj_kinematics(mj_model, data)
    return mujoco, model, mj_model, data, pose


def test_skeleton_matches_mujoco_bodies(case) -> None:
    mujoco, model, mj_model, data, pose = case
    body_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body) for body in range(1, mj_model.nbody)]
    assert model.joint_names == body_names
    assert model.parents == [int(parent) - 1 for parent in mj_model.body_parentid[1:]]

    skeleton = model.forward_skeleton(pose)
    rotations, positions = _in_model_frame(data, data.xpos[1:], data.xmat[1:].reshape(-1, 3, 3))
    np.testing.assert_allclose(skeleton[:, :3, 3], positions, atol=2e-5)
    np.testing.assert_allclose(skeleton[:, :3, :3], rotations, atol=2e-5)


def test_link_meshes_match_mujoco_visual_geoms(case) -> None:
    mujoco, model, mj_model, data, pose = case
    visual = [
        geom
        for geom in range(mj_model.ngeom)
        if mj_model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
        and mj_model.geom_contype[geom] == 0
        and mj_model.geom_conaffinity[geom] == 0
    ]
    assert len(visual) == len(model.link_names)

    for geom, mesh, link in zip(visual, model.link_meshes, model.forward_links(**{model._POSE: pose}), strict=True):
        mesh_id = mj_model.geom_dataid[geom]
        start = mj_model.mesh_vertadr[mesh_id]
        local = mj_model.mesh_vert[start : start + mj_model.mesh_vertnum[mesh_id]]
        world = local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        _, expected = _in_model_frame(data, world, np.eye(3))
        actual = np.asarray(mesh.vertices) @ link[:3, :3].T + link[:3, 3]
        np.testing.assert_allclose(actual.min(0), expected.min(0), atol=1e-4)
        np.testing.assert_allclose(actual.max(0), expected.max(0), atol=1e-4)


def test_coupled_hinges_satisfy_mujoco_equalities(case) -> None:
    mujoco, _, mj_model, data, _ = case
    for eq in range(mj_model.neq):
        assert mj_model.eq_type[eq] == mujoco.mjtEq.mjEQ_JOINT
        coupled = data.qpos[mj_model.jnt_qposadr[mj_model.eq_obj1id[eq]]]
        driver = data.qpos[mj_model.jnt_qposadr[mj_model.eq_obj2id[eq]]]
        expected = np.polynomial.polynomial.polyval(driver, mj_model.eq_data[eq, :5])
        np.testing.assert_allclose(coupled, expected, atol=1e-5)


def test_common_joints_and_presets_resolve(case) -> None:
    _, model, *_ = case
    for joint in model.common_joints:
        model.joint_index(joint)
    assert all(left != right for left, right in model.symmetric_joints)

    if isinstance(model, Hand):
        presets = [model.get_rest_pose(hands="flat"), model.get_rest_pose(hands="rest")]
    else:
        presets = [model.get_tpose(), model.get_apose()]
    lo, hi = np.asarray(model.actuated_joint_limits).T
    for params in presets:
        pose = params[model._POSE]
        assert np.all((pose >= lo - 1e-6) & (pose <= hi + 1e-6))


def _in_model_frame(data, positions, rotations):
    """Express MuJoCo world poses relative to the root body, in the Y-up model frame."""
    root_rot = data.xmat[1].reshape(3, 3)
    local_pos = (positions - data.xpos[1]) @ root_rot
    local_rot = root_rot.T @ rotations
    return MUJOCO_TO_MODEL @ local_rot @ MUJOCO_TO_MODEL.T, local_pos @ MUJOCO_TO_MODEL.T
