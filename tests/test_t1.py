import numpy as np
import pytest

from robot_models.t1._io import get_model_path
from robot_models.t1.numpy import T1


def test_t1_to_qpos_accepts_rest_pose_motion_dict() -> None:
    model = T1()
    motion = model.get_rest_pose(batch_dims=(2,), dtype=np.float32)

    qpos = model.to_qpos(**motion)

    assert qpos.shape == (2, 7 + model.num_dofs)
    np.testing.assert_array_equal(qpos[:, 7:], motion["body_pose"])


@pytest.mark.parametrize("convention", ["soma", "mujoco"])
def test_t1_forward_skeleton_matches_mujoco_fk(convention) -> None:
    mujoco = pytest.importorskip("mujoco")
    model = T1(convention=convention)
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
    mujoco_to_model = np.asarray(model._mujoco_to_model(), dtype=np.float32)
    body_ids = _t1_body_ids(mujoco, mj_model)

    assert list(body_ids) == model.joint_names
    for joint_index, body_id in enumerate(body_ids.values()):
        expected = mujoco_to_model @ data.xpos[body_id]
        np.testing.assert_allclose(skeleton[0, joint_index, :3, 3], expected, rtol=1e-6, atol=1e-6)


def test_t1_presets_match_rest_and_lowered_arms() -> None:
    model = T1()

    np.testing.assert_array_equal(model.get_tpose()["body_pose"], model.get_rest_pose()["body_pose"])
    apose = model.unpack_pose(model.get_apose(batch_dims=(2,))["body_pose"])
    np.testing.assert_allclose(apose["left_shoulder_roll_skel"], -np.pi / 4)
    np.testing.assert_allclose(apose["right_shoulder_roll_skel"], np.pi / 4)


def _t1_body_ids(mujoco, mj_model) -> dict[str, int]:
    body_ids = {"trunk_skel": mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")}
    for joint_id in range(1, mj_model.njnt):
        name = mj_model.joint(joint_id).name.lower().removeprefix("aa")
        body_ids[f"{name}_skel"] = int(mj_model.jnt_bodyid[joint_id])
    return body_ids
