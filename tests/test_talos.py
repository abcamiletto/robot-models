import numpy as np
import pytest

from robot_models import Joint
from robot_models.talos._io import get_model_path
from robot_models.talos.numpy import Talos


def test_talos_to_qpos_accepts_rest_pose_motion_dict() -> None:
    model = Talos()
    motion = model.get_rest_pose(batch_dims=(2,), dtype=np.float32)

    qpos = model.to_qpos(**motion)

    assert qpos.shape == (2, 7 + model.num_dofs)
    np.testing.assert_array_equal(qpos[:, 7:], motion["body_pose"])


def test_talos_tpose_raises_arms_sideways() -> None:
    model = Talos()
    skeleton = model.forward_skeleton(**model.get_tpose())

    for side, sign in (("left", 1.0), ("right", -1.0)):
        shoulder = skeleton[model.joint_index(Joint(f"{side}_shoulder")), :3, 3]
        wrist = skeleton[model.joint_index(Joint(f"{side}_wrist")), :3, 3]
        direction = (wrist - shoulder) / np.linalg.norm(wrist - shoulder)
        assert sign * direction[0] > 0.99


def test_talos_forward_skeleton_matches_mujoco_fk() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = Talos()
    mj_model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    data = mujoco.MjData(mj_model)
    body_pose = np.linspace(-0.1, 0.1, model.num_dofs, dtype=np.float32)[None]
    global_translation = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    global_rotation = np.array([[0.05, -0.1, 0.08]], dtype=np.float32)

    data.qpos[:] = model.to_qpos(
        body_pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
    )[0]
    mujoco.mj_forward(mj_model, data)
    skeleton = model.forward_skeleton(
        body_pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
    )
    mujoco_to_model = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    # Actuated joint k drives qpos[7 + k]; its MuJoCo body defines the skeleton joint frame.
    body_ids = {0: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")}
    for joint_id in range(mj_model.njnt):
        qpos_index = mj_model.jnt_qposadr[joint_id] - 7
        if qpos_index >= 0:
            name = model.actuated_joint_names[qpos_index]
            body_ids[model.joint_names.index(name)] = mj_model.jnt_bodyid[joint_id]
    assert len(body_ids) == model.num_dofs + 1

    for joint_index, body_id in body_ids.items():
        expected_pos = mujoco_to_model @ data.xpos[body_id]
        expected_rot = mujoco_to_model @ data.xmat[body_id].reshape(3, 3) @ mujoco_to_model.T
        np.testing.assert_allclose(skeleton[0, joint_index, :3, 3], expected_pos, rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(skeleton[0, joint_index, :3, :3], expected_rot, rtol=1e-6, atol=1e-5)

    for site in ("head", "left_foot", "right_foot"):
        site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site)
        expected = mujoco_to_model @ data.site_xpos[site_id]
        np.testing.assert_allclose(skeleton[0, model.joint_names.index(site), :3, 3], expected, rtol=1e-6, atol=1e-5)
