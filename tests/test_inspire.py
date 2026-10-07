import numpy as np
import pytest

from robot_models.inspire._io import get_model_path
from robot_models.inspire.numpy import InspireHand


@pytest.mark.parametrize("side", ["left", "right"])
def test_inspire_forward_skeleton_matches_mujoco_fk(side) -> None:
    mujoco = pytest.importorskip("mujoco")
    model = InspireHand(side=side)
    mj_model = mujoco.MjModel.from_xml_path(str(get_model_path() / f"{side}.xml"))
    data = mujoco.MjData(mj_model)
    limits = model.actuated_joint_limits
    hand_pose = (limits[:, 0] + 0.7 * (limits[:, 1] - limits[:, 0]))[None]

    for name, value in zip(model.actuated_joint_names, hand_pose[0], strict=True):
        data.joint(name.replace("_skel", "_joint")).qpos = value
    for equality_id in range(mj_model.neq):
        coupled = mj_model.eq_obj1id[equality_id]
        driver = mj_model.eq_obj2id[equality_id]
        coeffs = mj_model.eq_data[equality_id, :5]
        driver_q = data.qpos[mj_model.jnt_qposadr[driver]]
        data.qpos[mj_model.jnt_qposadr[coupled]] = np.polynomial.polynomial.polyval(driver_q, coeffs)
    mujoco.mj_forward(mj_model, data)
    skeleton = model.forward_skeleton(hand_pose)
    mujoco_to_model = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    assert mj_model.neq == 6
    for joint_index, joint_name in enumerate(model.joint_names):
        if joint_name == f"{side}_base_skel":
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_base_link")
        else:
            joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name.replace("_skel", "_joint"))
            body_id = mj_model.jnt_bodyid[joint_id]
        np.testing.assert_allclose(skeleton[0, joint_index, :3, 3], mujoco_to_model @ data.xpos[body_id], atol=1e-6)
        expected_rot = mujoco_to_model @ data.xmat[body_id].reshape(3, 3) @ mujoco_to_model.T
        np.testing.assert_allclose(skeleton[0, joint_index, :3, :3], expected_rot, atol=1e-5)


@pytest.mark.parametrize("hands", ["flat", "rest"])
def test_inspire_hand_presets(hands) -> None:
    model = InspireHand()
    params = model.get_rest_pose(batch_dims=(2,), hands=hands)

    assert params["hand_pose"].shape == (2, model.num_dofs)
    assert model.forward_skeleton(**params).shape == (2, model.num_joints, 4, 4)
