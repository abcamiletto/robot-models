"""Canonical hands mounted into MuJoCo scenes through the public helper."""

import subprocess
import sys

import model_cases
import numpy as np
import pytest

from robot_models import _assets as assets
from robot_models import add_hand
from robot_models._common.mjcf import MUJOCO_TO_MODEL
from robot_models.hands._base import Hand

mujoco = pytest.importorskip("mujoco")
CASES = [(name, side) for name, cls, _ in model_cases.MODELS if issubclass(cls, Hand) for side in cls.supported_sides]


@pytest.mark.parametrize("name,side", CASES)
def test_mounted_hand_matches_fk_and_preserves_definitions(name, side):
    hand = model_cases.backend_model_class(name, "numpy")(side=side)
    directory = assets.model_dir(name)
    original = mujoco.MjModel.from_xml_path(str(directory / f"{side}.xml"))
    scene = mujoco.MjSpec()
    scene.option.timestep = 0.003
    arm = scene.worldbody.add_body(name="arm", pos=[0.3, -0.2, 0.4])
    arm.add_joint(name="arm_rotation", axis=[0, 0, 1])
    arm.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], mass=1)
    frame = arm.add_frame(name="wrist", pos=[0.1, 0.2, 0.3], quat=[0.5, 0.5, -0.5, 0.5])
    attachment = add_hand(scene, name, side=side, frame=frame, model_path=directory)
    attachment.pos = [0.04, 0.03, -0.02]
    model = scene.compile()
    data = mujoco.MjData(model)
    prefix = f"{name}_{side}_"
    indices = [model.body(prefix + joint).id for joint in hand.joint_names]
    for attribute in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
        np.testing.assert_allclose(getattr(model, attribute)[indices], getattr(original, attribute)[1:], atol=1e-12)
    assert (model.nu, model.neq, model.ntendon, model.nexclude, model.npair) == (
        original.nu,
        original.neq,
        original.ntendon,
        original.nexclude,
        original.npair,
    )
    assert model.ngeom == original.ngeom + 1
    assert model.nv == len(hand.to_qpos(np.zeros(hand.num_dofs))[7:]) + 1
    assert model.opt.timestep == 0.003

    lo, hi = np.asarray(hand.actuated_joint_limits).T
    for fraction in (0.0, 0.37, 1.0):
        pose = (lo + fraction * (hi - lo)).astype(np.float32)
        data.qpos[0] = 0.63
        data.qpos[1:] = hand.to_qpos(pose)[7:]
        mujoco.mj_kinematics(model, data)
        # Express the mounted result in its wrist frame, then change Z-up to Y-up.
        root_rotation = data.xmat[indices[0]].reshape(3, 3)
        positions = (data.xpos[indices] - data.xpos[indices[0]]) @ root_rotation
        rotations = root_rotation.T @ data.xmat[indices].reshape(-1, 3, 3)
        expected = hand.forward_skeleton(pose)
        np.testing.assert_allclose(positions @ MUJOCO_TO_MODEL.T, expected[:, :3, 3], atol=2e-6)
        np.testing.assert_allclose(MUJOCO_TO_MODEL @ rotations @ MUJOCO_TO_MODEL.T, expected[:, :3, :3], atol=2e-6)
        # The mounting transform belongs to the scene and follows its articulated arm.
        arm_rotation = data.xmat[model.body("arm").id].reshape(3, 3)
        frame_rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        expected_position = data.xpos[model.body("arm").id] + arm_rotation @ (
            np.array([0.1, 0.2, 0.3]) + frame_rotation @ attachment.pos
        )
        np.testing.assert_allclose(data.xpos[indices[0]], expected_position, atol=1e-12)
        np.testing.assert_allclose(root_rotation, arm_rotation @ frame_rotation, atol=1e-12)


def test_multiple_hands_use_separate_namespaces():
    scene = mujoco.MjSpec()
    add_hand(scene, "allegro", side="left")
    add_hand(scene, "allegro", side="right")
    add_hand(scene, "allegro", side="right", prefix="extra_")
    model = scene.compile()
    for prefix in ("allegro_left_", "allegro_right_", "extra_"):
        assert model.body(prefix + "palm").id > 0


@pytest.mark.parametrize("name", ["barrett", "robotiq-3finger", "ezgripper"])
def test_mounted_hand_filters_self_contacts_and_keeps_object_contacts(name):
    scene = mujoco.MjSpec()
    add_hand(scene, name)
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    (collider,) = np.flatnonzero(model.geom_contype | model.geom_conaffinity)[:1]
    object_body = scene.worldbody.add_body(pos=data.geom_xpos[collider])
    object_body.add_freejoint()
    object_body.add_geom(name="object", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0])
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    object_id = model.geom("object").id
    assert any(object_id in (contact.geom1, contact.geom2) for contact in data.contact)


@pytest.mark.fast
@pytest.mark.parametrize(
    "name,side,error", [("g1", "right", TypeError), ("unknown", "right", ValueError), ("barrett", "left", ValueError)]
)
def test_invalid_hand_does_not_change_scene(name, side, error, monkeypatch):
    def unexpected_download(*args, **kwargs):
        pytest.fail("Invalid hand must fail before resolving assets")

    monkeypatch.setattr(assets, "model_dir", unexpected_download)
    scene = mujoco.MjSpec()
    with pytest.raises(error):
        add_hand(scene, name, side=side)
    assert not scene.worldbody.bodies
    assert not scene.worldbody.frames


@pytest.mark.fast
def test_mujoco_is_an_optional_lazy_import():
    code = "import sys; import robot_models; assert 'mujoco' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
