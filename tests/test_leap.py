import numpy as np
import pytest
from nanomanifold import SO3

from robot_models.leap._io import get_model_path
from robot_models.leap.numpy import LeapHand


@pytest.mark.parametrize("side", ["left", "right"])
def test_leap_forward_skeleton_matches_mujoco_fk(side) -> None:
    mujoco = pytest.importorskip("mujoco")
    model = LeapHand(side=side)
    mj_model = mujoco.MjModel.from_xml_path(str(get_model_path() / f"{side}.xml"))
    data = mujoco.MjData(mj_model)
    hand_pose = np.linspace(-0.3, 0.6, model.num_dofs, dtype=np.float32)[None]
    global_translation = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    global_rotation = np.array([[0.05, -0.1, 0.08]], dtype=np.float32)

    data.qpos[:] = hand_pose[0]
    mujoco.mj_forward(mj_model, data)
    skeleton = model.forward_skeleton(
        hand_pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
    )
    mujoco_to_model = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    global_rot = SO3.convert(global_rotation[0], src="axis_angle", dst="rotmat", xp=np)
    palm_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "palm")

    for joint_index, joint_name in enumerate(model.joint_names):
        native_name = joint_name.removeprefix(f"{side}_").removesuffix("_skel")
        if native_name == "palm":
            body_id = palm_id
        else:
            body_id = mj_model.jnt_bodyid[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, native_name)]
        position = mujoco_to_model @ (data.xpos[body_id] - data.xpos[palm_id])
        rotation = mujoco_to_model @ data.xmat[body_id].reshape(3, 3) @ mujoco_to_model.T
        np.testing.assert_allclose(
            skeleton[0, joint_index, :3, 3],
            global_rot @ position + global_translation[0],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(skeleton[0, joint_index, :3, :3], global_rot @ rotation, rtol=1e-5, atol=1e-5)
