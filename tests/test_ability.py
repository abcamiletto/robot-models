import numpy as np
import pytest

from robot_models.ability._io import get_model_path
from robot_models.ability.numpy import AbilityHand


@pytest.mark.parametrize("side", ["left", "right"])
def test_ability_forward_skeleton_matches_mujoco_fk(side) -> None:
    mujoco = pytest.importorskip("mujoco")
    model = AbilityHand(side=side)
    mj_model = mujoco.MjModel.from_xml_path(str(get_model_path() / f"{side}.xml"))
    data = mujoco.MjData(mj_model)
    limits = model.actuated_joint_limits
    hand_pose = (limits[:, 0] + np.linspace(0.2, 0.8, model.num_dofs) * (limits[:, 1] - limits[:, 0]))[None]

    def joint_id(skel_name: str) -> int:
        return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, skel_name.replace("_skel", "_joint"))

    for name, value in zip(model.actuated_joint_names, hand_pose[0], strict=True):
        data.qpos[mj_model.jnt_qposadr[joint_id(name)]] = value
    for eq_id in range(mj_model.neq):
        coupled, driver = mj_model.eq_obj1id[eq_id], mj_model.eq_obj2id[eq_id]
        driver_pose = data.qpos[mj_model.jnt_qposadr[driver]]
        data.qpos[mj_model.jnt_qposadr[coupled]] = np.polyval(mj_model.eq_data[eq_id, :5][::-1], driver_pose)
    mujoco.mj_forward(mj_model, data)
    skeleton = model.forward_skeleton(hand_pose)
    mujoco_to_model = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    equality = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    np.testing.assert_allclose(data.efc_pos[equality], 0.0, atol=1e-6)
    for joint_index, joint_name in enumerate(model.joint_names):
        if joint_name == f"{side}_base_skel":
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_base_link")
        else:
            body_id = mj_model.jnt_bodyid[joint_id(joint_name)]
        expected_pos = mujoco_to_model @ data.xpos[body_id]
        expected_rot = mujoco_to_model @ data.xmat[body_id].reshape(3, 3) @ mujoco_to_model.T
        np.testing.assert_allclose(skeleton[0, joint_index, :3, 3], expected_pos, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(skeleton[0, joint_index, :3, :3], expected_rot, rtol=1e-5, atol=1e-5)


def test_ability_rest_presets_have_dof_shape() -> None:
    model = AbilityHand()
    for hands in ("flat", "rest"):
        params = model.get_rest_pose(batch_dims=(2,), hands=hands)
        assert params["hand_pose"].shape == (2, model.num_dofs)
