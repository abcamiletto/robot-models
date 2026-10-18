"""Canonical hands mounted into MuJoCo scenes through the public helper."""

import subprocess
import sys

import model_cases
import numpy as np
import pytest

from robot_models import _assets as assets
from robot_models import _catalog as catalog
from robot_models import add_hand
from robot_models._common.mjcf import MUJOCO_TO_MODEL

mujoco = pytest.importorskip("mujoco")
CASES = [(name, side) for name, spec in catalog.MODEL_SPECS.items() for side in spec.sides]


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
    attachment, root = add_hand(scene, name, side=side, frame=frame, model_path=directory)
    attachment.pos = [0.04, 0.03, -0.02]
    model = scene.compile()
    data = mujoco.MjData(model)
    prefix = f"{name}_{side}_"
    assert root.name == prefix + hand.joint_names[0]
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
    if not np.isfinite(hand.effort_limits).any():
        np.testing.assert_array_equal(model.actuator_forcerange, original.actuator_forcerange)
        np.testing.assert_array_equal(model.actuator_forcelimited, original.actuator_forcelimited)

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


@pytest.mark.fast
@pytest.mark.parametrize(
    "frame_name,free,match",
    [
        (None, True, "world-level"),
        ("wrist", True, "world-level"),
        ("missing", False, "not found"),
        ("missing", True, "not found"),
    ],
)
def test_invalid_mount_fails_before_resolving_assets(frame_name, free, match, monkeypatch):
    def unexpected_download(*args, **kwargs):
        pytest.fail("Invalid mount must fail before resolving assets")

    monkeypatch.setattr(assets, "model_dir", unexpected_download)
    scene = mujoco.MjSpec()
    arm = scene.worldbody.add_body(name="arm")
    frame = arm.add_frame(name="wrist")
    with pytest.raises(ValueError, match=match):
        add_hand(scene, "shadow", free=free, frame=frame if frame_name is None else frame_name)
    assert [body.name for body in scene.bodies] == ["world", "arm"]
    assert not arm.bodies
    assert not arm.joints


@pytest.mark.parametrize("name,side", CASES)
def test_free_mount_is_quiet_and_starts_with_root_coordinates(name, side, tmp_path, monkeypatch):
    import warnings

    monkeypatch.chdir(tmp_path)
    scene = mujoco.MjSpec()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame, root = add_hand(scene, name, side=side, free=True)
        frame.pos = [0.1, -0.2, 0.4]
        frame.quat = [0.5, 0.5, 0.5, 0.5]
        model = scene.compile()
    assert not (tmp_path / "MUJOCO_LOG.TXT").exists()
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.jnt_bodyid[0] == model.body(root.name).id
    assert model.jnt_qposadr[0] == 0
    np.testing.assert_allclose(model.qpos0[:7], [0.1, -0.2, 0.4, 0.5, 0.5, 0.5, 0.5])
    if name in ("allegro", "leap"):
        limit = 0.7 if name == "allegro" else 0.93
        assert model.actuator_forcelimited.all()
        np.testing.assert_allclose(model.actuator_forcerange, np.tile([-limit, limit], (model.nu, 1)))
        data = mujoco.MjData(model)
        data.ctrl[:] = 1e6
        mujoco.mj_forward(model, data)
        assert np.all(np.abs(data.actuator_force) <= limit + 1e-6)


def test_effort_limits_account_for_gear_and_preserve_stricter_bounds(tmp_path):
    xml = """<mujoco><worldbody><body name="palm">
      <joint name="ffj1" type="hinge"/>
      <geom type="sphere" size="0.01" mass="1"/>
      </body></worldbody><actuator>
      <motor name="geared" joint="ffj1" gear="2"/>
      <motor name="stricter" joint="ffj1" gear="-2" forcerange="-0.1 0.2"/>
      </actuator></mujoco>"""
    (tmp_path / "right.xml").write_text(xml)
    scene = mujoco.MjSpec()
    add_hand(scene, "allegro", model_path=tmp_path)
    model = scene.compile()
    np.testing.assert_allclose(model.actuator_forcerange, [[-0.35, 0.35], [-0.1, 0.2]])


@pytest.mark.parametrize("name,side", CASES)
def test_qpos_places_the_hand_in_any_scene_frame_by_joint_names(name, side):
    hand = model_cases.backend_model_class(name, "numpy")(side=side)
    scene = mujoco.MjSpec()
    _, root = add_hand(scene, name, side=side, prefix="h1-", free=True)
    model = scene.compile()
    data = mujoco.MjData(model)
    assert len(hand.qpos_joint_names) == model.nq - 7

    rng = np.random.default_rng(0)
    lo, hi = np.asarray(hand.actuated_joint_limits).T
    pose = rng.uniform(np.maximum(lo, -1.0), np.minimum(hi, 1.0)).astype(np.float32)
    rotation, translation = rng.normal(size=3), rng.normal(size=3)
    scene_transform = np.eye(4)
    scene_transform[:3, :3] = _rotation(rng)
    scene_transform[:3, 3] = rng.normal(size=3)
    qpos = hand.to_qpos(pose, global_rotation=rotation, global_translation=translation, scene_transform=scene_transform)
    root_address = model.jnt_qposadr[model.body_jntadr[model.body(root.name).id]]
    data.qpos[root_address : root_address + 7] = qpos[:7]
    for joint, angle in zip(hand.qpos_joint_names, qpos[7:], strict=True):
        data.qpos[model.joint("h1-" + joint).qposadr[0]] = angle
    mujoco.mj_kinematics(model, data)

    frames = hand.forward_skeleton(pose, global_rotation=rotation, global_translation=translation)
    bodies = [model.body("h1-" + joint).id for joint in hand.joint_names]
    rotation_to_scene, offset = scene_transform[:3, :3], scene_transform[:3, 3]
    np.testing.assert_allclose(data.xpos[bodies], frames[:, :3, 3] @ rotation_to_scene.T + offset, atol=2e-5)
    expected = rotation_to_scene @ frames[:, :3, :3] @ MUJOCO_TO_MODEL
    np.testing.assert_allclose(data.xmat[bodies].reshape(-1, 3, 3), expected, atol=2e-5)


def _rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q * np.linalg.det(q)
