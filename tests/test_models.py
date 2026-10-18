"""Every model against MuJoCo's own reading of its MJCF file."""

import model_cases
import numpy as np
import pytest

from robot_models import Joint
from robot_models import _catalog as catalog
from robot_models._common.mjcf import MUJOCO_TO_MODEL
from robot_models.hands._base import Hand

CASES = [
    (name, kwargs)
    for name, model_class, _ in model_cases.MODELS
    for kwargs in (
        [{"side": side} for side in catalog.MODEL_SPECS[name].sides] if issubclass(model_class, Hand) else [{}]
    )
]


@pytest.fixture(params=CASES, ids=lambda case: "-".join(filter(None, (case[0], case[1].get("side")))))
def case(request):
    """A model, MuJoCo's model of the same file, and both posed with one random pose."""
    mujoco = pytest.importorskip("mujoco")
    name, kwargs = request.param
    model = model_cases.backend_model_class(name, "numpy")(**kwargs)
    xml_path = model_cases.xml_path(name, kwargs)
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

    for eq in range(mj_model.neq):
        if mj_model.eq_type[eq] != mujoco.mjtEq.mjEQ_JOINT:
            continue  # Tendon equalities describe simulation rather than hinge pose coordinates.
        coupled = data.qpos[mj_model.jnt_qposadr[mj_model.eq_obj1id[eq]]]
        driver = data.qpos[mj_model.jnt_qposadr[mj_model.eq_obj2id[eq]]]
        expected = np.polynomial.polynomial.polyval(driver, mj_model.eq_data[eq, :5])
        np.testing.assert_allclose(coupled, expected, atol=1e-5)


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


def test_collision_meshes_match_mujoco_geoms(case) -> None:
    mujoco, model, mj_model, data, pose = case
    paired = set(mj_model.pair_geom1) | set(mj_model.pair_geom2)
    geoms = [
        i for i in range(mj_model.ngeom) if mj_model.geom_contype[i] or mj_model.geom_conaffinity[i] or i in paired
    ]
    assert geoms, "Every model must supply collision geometry"
    if model.name == "shadow":
        assert len(geoms) == 38
    assert len(model.collision_meshes) == len(geoms) == len(model.collision_names)
    transforms = model.forward_links(geometry="collision", **{model._POSE: pose})
    directions = np.vstack((np.eye(3), -np.eye(3), np.random.default_rng(4).normal(size=(24, 3))))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    vertices = []
    for geom, mesh, transform, joint in zip(
        geoms, model.collision_meshes, transforms, model.collision_joint_indices, strict=True
    ):
        assert model.joint_names[joint] == mj_model.body(int(mj_model.geom_bodyid[geom])).name
        world = np.asarray(mesh.vertices) @ transform[:3, :3].T + transform[:3, 3]
        vertices.append(world)
        rotation, position = _in_model_frame(data, data.geom_xpos[geom], data.geom_xmat[geom].reshape(3, 3))
        local = (world - position) @ rotation @ MUJOCO_TO_MODEL
        kind = mj_model.geom_type[geom]
        size = mj_model.geom_size[geom]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            index = mj_model.geom_dataid[geom]
            start, count = mj_model.mesh_vertadr[index], mj_model.mesh_vertnum[index]
            expected = mj_model.mesh_vert[start : start + count]
            np.testing.assert_allclose((local @ directions.T).max(0), (expected @ directions.T).max(0), atol=2e-6)
            continue
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            distance = (np.abs(local) - size).max(1)
            extent = size
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            distance = np.linalg.norm(local, axis=1) - size[0]
            extent = np.repeat(size[0], 3)
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            distance = np.maximum(np.linalg.norm(local[:, :2], axis=1) - size[0], np.abs(local[:, 2]) - size[1])
            extent = [size[0], size[0], size[1]]
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            local_axis = local.copy()
            local_axis[:, 2] -= np.clip(local_axis[:, 2], -size[1], size[1])
            distance = np.linalg.norm(local_axis, axis=1) - size[0]
            extent = [size[0], size[0], size[1] + size[0]]
        else:
            assert kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID
            distance = np.linalg.norm(local / size, axis=1) - 1
            extent = size
        np.testing.assert_allclose(distance, 0, atol=2e-6)
        # Curved primitives are tessellated; the native MuJoCo shapes stay analytic.
        np.testing.assert_allclose(np.abs(local).max(0), extent, rtol=0.003, atol=2e-6)
    posed = model.forward_meshes(geometry="collision", **{model._POSE: pose})[0]
    np.testing.assert_allclose(posed.vertices, np.concatenate(vertices), atol=2e-6)


def test_common_joints_and_presets_resolve(case) -> None:
    _, model, *_ = case
    for joint in model.common_joints:
        model.joint_index(joint)
    assert all(left != right for left, right in model.symmetric_joints)
    for joint in model.common_joints:
        if joint.value.startswith("left_"):
            counterpart = Joint(joint.value.replace("left_", "right_", 1))
            if counterpart in model.common_joints:
                pair = (model.joint_index(joint), model.joint_index(counterpart))
                assert pair in model.symmetric_joints

    if isinstance(model, Hand):
        presets = [model.get_rest_pose(hands=preset) for preset in ("flat", "rest", "open", "closed")]
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
