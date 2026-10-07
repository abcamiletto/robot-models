import numpy as np
import pytest

from robot_models.shadow._io import get_model_path
from robot_models.shadow.numpy import ShadowHand

MUJOCO_TO_MODEL = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)


def _mujoco_pose(side: str):
    mujoco = pytest.importorskip("mujoco")
    model = ShadowHand(side=side)
    mj_model = mujoco.MjModel.from_xml_path(str(get_model_path() / f"{side}.xml"))
    data = mujoco.MjData(mj_model)
    limits = model.actuated_joint_limits
    hand_pose = (0.3 * limits[:, 0] + 0.7 * limits[:, 1])[None]
    data.qpos[:] = hand_pose[0]
    mujoco.mj_forward(mj_model, data)
    return mujoco, model, mj_model, data, hand_pose


@pytest.mark.parametrize("side", ["left", "right"])
def test_shadow_forward_skeleton_matches_mujoco_fk(side) -> None:
    mujoco, model, mj_model, data, hand_pose = _mujoco_pose(side)
    skeleton = model.forward_skeleton(hand_pose)

    assert model.num_dofs == mj_model.nq == 24
    for joint_index, joint_name in enumerate(model.joint_names):
        body_name = f"{side[0]}h_{joint_name.removeprefix(f'{side}_').removesuffix('_skel')}"
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        expected_rot = MUJOCO_TO_MODEL @ data.xmat[body_id].reshape(3, 3) @ MUJOCO_TO_MODEL.T
        np.testing.assert_allclose(skeleton[0, joint_index, :3, 3], MUJOCO_TO_MODEL @ data.xpos[body_id], atol=1e-6)
        np.testing.assert_allclose(skeleton[0, joint_index, :3, :3], expected_rot, atol=1e-5)


@pytest.mark.parametrize("side", ["left", "right"])
def test_shadow_link_meshes_match_mujoco_geoms(side) -> None:
    _, model, mj_model, data, hand_pose = _mujoco_pose(side)
    links = model.forward_links(hand_pose=hand_pose)[0]
    visual_geoms = [g for g in range(mj_model.ngeom) if mj_model.geom_group[g] == 2]

    assert len(visual_geoms) == len(model.link_meshes)
    for link, geom, mesh in zip(links, visual_geoms, model.link_meshes, strict=True):
        mesh_id = mj_model.geom_dataid[geom]
        start = mj_model.mesh_vertadr[mesh_id]
        local = mj_model.mesh_vert[start : start + mj_model.mesh_vertnum[mesh_id]]
        expected = (data.geom_xpos[geom] + local @ data.geom_xmat[geom].reshape(3, 3).T) @ MUJOCO_TO_MODEL.T
        actual = mesh.vertices @ link[:3, :3].T + link[:3, 3]
        np.testing.assert_allclose(actual.min(axis=0), expected.min(axis=0), atol=1e-5)
        np.testing.assert_allclose(actual.max(axis=0), expected.max(axis=0), atol=1e-5)


@pytest.mark.parametrize("side", ["left", "right"])
def test_shadow_hand_presets(side) -> None:
    model = ShadowHand(side=side)
    for hands in ("flat", "rest"):
        params = model.get_rest_pose(batch_dims=(2,), hands=hands)
        assert params["hand_pose"].shape == (2, model.num_dofs)
    for joint in model.common_joints:
        assert model.joint_names[model.joint_index(joint)].startswith(f"{side}_")
