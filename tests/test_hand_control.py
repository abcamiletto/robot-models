"""Hand semantics and closing motion against native MuJoCo geometry."""

import numpy as np
import pytest

from robot_models import _assets as assets
from robot_models import _catalog as catalog
from robot_models import create_model

CASES = [(name, side) for name, spec in catalog.MODEL_SPECS.items() for side in spec.sides]
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# Palm body, palmar face (axis/sign in that body's frame), and distal bodies in finger order.
POINTS = {
    "ability": (
        "{side}_base_link",
        None,
        ["{side}_thumb_proximal_link", *["{side}_" + f + "_distal_link" for f in FINGERS[1:]]],
    ),
    "allegro": ("palm", (0, 1), ["th_tip", "ff_tip", "mf_tip", "rf_tip"]),
    "barrett": ("bh_base_link", (2, 1), ["bh_finger_33_link", "bh_finger_13_link", "bh_finger_23_link"]),
    "brainco": ("{side}_base_link", None, ["{side}_" + f + "_distal_link" for f in FINGERS]),
    "ezgripper": ("mount", None, ["F1_L2", "F2_L2"]),
    "inspire": (
        "{side}_hand_base_link",
        None,
        ["{side}_thumb_distal", *["{side}_" + f + "_intermediate" for f in FINGERS[1:]]],
    ),
    "leap": ("palm", (2, -1), ["th_ds", "if_ds", "mf_ds", "rf_ds"]),
    "robotiq-3finger": ("palm", (1, 1), ["finger_middle_link_3", "finger_1_link_3", "finger_2_link_3"]),
    "shadow": ("{s}h_palm", (1, -1), ["{s}h_" + f + "distal" for f in ("th", "ff", "mf", "rf", "lf")]),
    "sharpa": ("{side}_hand_C_MC", (0, 1), ["{side}_" + f + "_DP" for f in FINGERS]),
}


@pytest.mark.parametrize("name,side", CASES)
def test_closing_semantics_match_native_finger_motion(name, side):
    mujoco = pytest.importorskip("mujoco")
    hand = create_model(name, side=side)
    assert len(hand.dofs) == len(hand.actuated_joint_names)
    assert hand.effort_limits.shape == (hand.num_dofs,)
    expected_effort = {"allegro": 0.7, "leap": 0.93}.get(name, np.inf)
    np.testing.assert_allclose(hand.effort_limits, expected_effort)
    opened = hand.get_rest_pose(hands="open")["hand_pose"]
    closed = hand.get_rest_pose(hands="closed")["hand_pose"]
    np.testing.assert_allclose(hand.close(opened, 1000), closed)
    np.testing.assert_array_equal(hand.close(opened, 0), opened)

    model = mujoco.MjModel.from_xml_path(str(assets.model_dir(name) / f"{side}.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = hand.to_qpos(opened)[-model.nq :]
    mujoco.mj_kinematics(model, data)
    palm_name, face, tip_names = POINTS[name]
    palm = model.body(palm_name.format(side=side, s=side[0])).id
    cloud = (_mesh_points(model, data, palm) - data.xpos[palm]) @ data.xmat[palm].reshape(3, 3)
    palm_point = (cloud.min(0) + cloud.max(0)) / 2
    if face:
        axis, sign = face
        palm_point[axis] = cloud[:, axis].max() if sign > 0 else cloud[:, axis].min()
    palm_world = palm_point @ data.xmat[palm].reshape(3, 3).T + data.xpos[palm]
    bodies, local = [], []
    for tip_name in tip_names:
        body = model.body(tip_name.format(side=side, s=side[0])).id
        cloud = _mesh_points(model, data, body)
        # Keep a fixed point on the distal end of the original visual surface.
        radii = np.linalg.norm(cloud - palm_world, axis=1)
        point = cloud[radii >= np.quantile(radii, 0.98)].mean(0)
        bodies.append(body)
        local.append((point - data.xpos[body]) @ data.xmat[body].reshape(3, 3))

    def tips(pose):
        data.qpos[:] = hand.to_qpos(pose)[-model.nq :]
        mujoco.mj_kinematics(model, data)
        return np.array(
            [point @ data.xmat[body].reshape(3, 3).T + data.xpos[body] for body, point in zip(bodies, local)]
        )

    initial = tips(opened)
    moved = tips(hand.close(opened, 0.02))
    assert np.all(np.linalg.norm(moved - palm_world, axis=1) < np.linalg.norm(initial - palm_world, axis=1))
    lower, upper = np.asarray(hand.actuated_joint_limits).T
    for index, dof in enumerate(hand.dofs):
        if dof.finger not in FINGERS:
            continue
        pose = opened.copy()
        pose[index] += 0.02 if pose[index] + 0.02 <= upper[index] else -0.02
        pose = np.clip(pose, lower, upper)
        displacement = np.linalg.norm(tips(pose) - initial, axis=1)
        finger_index = FINGERS.index(dof.finger)
        assert displacement[finger_index] > 1e-7, hand.actuated_joint_names[index]
        np.testing.assert_allclose(np.delete(displacement, finger_index), 0, atol=1e-7)


@pytest.mark.parametrize("name,side", [("shadow", "left"), ("robotiq-3finger", "right")])
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_close_broadcasts_and_preserves_backend(name, side, backend):
    if backend != "numpy":
        pytest.importorskip(backend)
    reference = create_model(name, side=side)
    pose = reference.get_rest_pose(hands="open", batch_dims=(2, 3))["hand_pose"].copy()
    angles = np.array([[0, 0.02, 10], [0.03, -0.02, -10]], dtype=np.float32)
    expected = reference.close(pose, angles)
    hand = create_model(name, side=side, runtime=backend)
    pose = hand.runtime.asarray(pose, like=hand.actuated_joint_limits)
    amount = hand.runtime.asarray(angles, like=pose)
    actual = hand.close(pose, amount)
    assert type(actual) is type(pose) and actual.dtype == pose.dtype
    np.testing.assert_allclose(hand.runtime.to_numpy(actual), expected, atol=1e-6)
    if backend == "jax":
        import jax

        np.testing.assert_allclose(jax.jit(hand.close)(pose, amount), expected, atol=1e-6)
        gradient = jax.grad(lambda value: hand.close(value, 0.01).sum())(pose)
        np.testing.assert_allclose(gradient, 1)
    if backend == "torch":
        pose = pose.clone().requires_grad_(True)
        hand.close(pose, 0.01).sum().backward()
        assert pose.grad is not None
        np.testing.assert_allclose(pose.grad.detach().numpy(), 1)


def _mesh_points(model, data, body):
    import mujoco

    pieces = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body or model.geom_contype[geom] or model.geom_conaffinity[geom]:
            continue
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start : start + count]
        pieces.append(vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
    return np.concatenate(pieces) if pieces else data.xpos[[body]]
