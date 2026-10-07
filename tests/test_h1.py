import numpy as np
import pytest

from robot_models.h1._io import get_model_path
from robot_models.h1.numpy import H1


def test_h1_to_qpos_accepts_rest_pose_motion_dict() -> None:
    model = H1()
    motion = model.get_rest_pose(batch_dims=(2,), dtype=np.float32)

    qpos = model.to_qpos(**motion)

    assert qpos.shape == (2, 7 + model.num_dofs)
    np.testing.assert_array_equal(qpos[:, 7:], motion["body_pose"])


@pytest.mark.parametrize("convention", ["soma", "mujoco"])
def test_h1_presets_do_not_depend_on_convention(convention) -> None:
    model = H1(convention=convention)

    tpose = model.get_tpose(batch_dims=(2,))["body_pose"]
    apose = model.get_apose()["body_pose"]

    assert tpose.shape == (2, model.num_dofs)
    assert tpose[0, model.actuated_joint_names.index("left_shoulder_roll_skel")] == pytest.approx(np.pi / 2)
    assert apose[model.actuated_joint_names.index("right_shoulder_roll_skel")] == pytest.approx(-np.pi / 4)


def test_h1_forward_skeleton_matches_mujoco_fk() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = H1()
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

    for joint_index, joint_name in enumerate(model.joint_names):
        body_name = joint_name.replace("_skel", "" if joint_name == "pelvis_skel" else "_link")
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        expected_rot = mujoco_to_model @ data.xmat[body_id].reshape(3, 3) @ mujoco_to_model.T
        np.testing.assert_allclose(skeleton[0, joint_index, :3, 3], mujoco_to_model @ data.xpos[body_id], atol=1e-6)
        np.testing.assert_allclose(skeleton[0, joint_index, :3, :3], expected_rot, atol=1e-5)
