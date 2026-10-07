import numpy as np
import pytest

from robot_models.gr1._io import get_model_path
from robot_models.gr1.numpy import GR1


def test_gr1_to_qpos_accepts_rest_pose_motion_dict() -> None:
    model = GR1()
    motion = model.get_rest_pose(batch_dims=(2,), dtype=np.float32)

    qpos = model.to_qpos(**motion)

    assert qpos.shape == (2, 7 + model.num_dofs)
    np.testing.assert_array_equal(qpos[:, 7:], motion["body_pose"])


def test_gr1_forward_skeleton_matches_mujoco_fk() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = GR1()
    mj_model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    data = mujoco.MjData(mj_model)
    body_pose = np.linspace(-0.3, 0.3, model.num_dofs, dtype=np.float32)[None]
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
        if joint_name == "base_skel":
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        else:
            body_id = mj_model.joint(joint_name.replace("_skel", "_joint")).bodyid[0]
        expected = mujoco_to_model @ data.xpos[body_id]
        np.testing.assert_allclose(skeleton[0, joint_index, :3, 3], expected, rtol=1e-6, atol=1e-5)


def test_gr1_tpose_holds_arms_horizontal() -> None:
    model = GR1()
    skeleton = model.forward_skeleton(**model.get_tpose())
    for side in ("left", "right"):
        elbow = skeleton[model.joint_names.index(f"{side}_elbow_pitch_skel"), :3, 3]
        wrist = skeleton[model.joint_names.index(f"{side}_wrist_roll_skel"), :3, 3]
        np.testing.assert_allclose(wrist[1], elbow[1], atol=1e-3)
        assert abs(wrist[0]) > abs(elbow[0]) + 0.2
