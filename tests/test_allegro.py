import numpy as np
import pytest

from robot_models.allegro._io import get_model_path
from robot_models.allegro.numpy import AllegroHand


@pytest.mark.parametrize("hands", ["flat", "rest"])
def test_allegro_hand_presets_are_within_limits(hands) -> None:
    for side in ("left", "right"):
        model = AllegroHand(side=side)
        hand_pose = model.get_rest_pose(batch_dims=(2,), hands=hands)["hand_pose"]
        limits = model.actuated_joint_limits

        assert hand_pose.shape == (2, 16)
        assert np.all((hand_pose >= limits[:, 0]) & (hand_pose <= limits[:, 1]))


@pytest.mark.parametrize("side", ["left", "right"])
def test_allegro_forward_skeleton_matches_mujoco_fk(side) -> None:
    mujoco = pytest.importorskip("mujoco")
    model = AllegroHand(side=side)
    limits = model.actuated_joint_limits
    hand_pose = (limits[:, 0] + np.linspace(0.2, 0.8, model.num_dofs) * (limits[:, 1] - limits[:, 0]))[None]
    global_translation = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    global_rotation = np.array([[0.05, -0.1, 0.08]], dtype=np.float32)
    qpos = model.to_qpos(
        hand_pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
    )[0].astype(np.float64)

    # The upstream XML has no free joint, and the palm keeps its own rest rotation.
    # Apply the root part of qpos to the palm pose instead.
    spec = mujoco.MjSpec.from_file(str(get_model_path() / f"{side}.xml"))
    palm = spec.worldbody.first_body()
    palm_quat = np.zeros(4)
    mujoco.mju_mulQuat(palm_quat, qpos[3:7], palm.quat / np.linalg.norm(palm.quat))
    palm.pos = qpos[:3]
    palm.quat = palm_quat
    mj_model = spec.compile()
    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos[7:]
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

    for joint_index, joint_name in enumerate(model.joint_names):
        body_name = joint_name.removeprefix(f"{side}_").removesuffix("_skel")
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        expected = mujoco_to_model @ data.xpos[body_id]
        expected_rot = mujoco_to_model @ data.xmat[body_id].reshape(3, 3) @ mujoco_to_model.T
        np.testing.assert_allclose(skeleton[0, joint_index, :3, 3], expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(skeleton[0, joint_index, :3, :3], expected_rot, rtol=1e-5, atol=1e-5)
