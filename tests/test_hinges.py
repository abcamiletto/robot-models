"""Bodies with several hinges and inline meshes against MuJoCo."""

import numpy as np
import pytest

from robot_models._common import mjcf, rigid
from robot_models._common.mjcf import MUJOCO_TO_MODEL


@pytest.mark.fast
def test_hinges_of_one_body_compose_in_order_as_in_mujoco(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    path = tmp_path / "chain.xml"
    path.write_text("""<mujoco>
      <compiler angle="radian"/>
      <asset><mesh name="tetra" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1" face="0 2 1  0 1 3  0 3 2  1 2 3"/></asset>
      <worldbody><body name="root">
        <body name="ball" pos="0.1 0.2 0.3" euler="0.3 -0.2 0.1">
          <joint name="a" axis="0 0 1"/><joint name="b" axis="0 1 1"/><joint name="c" axis="1 0 0"/>
          <geom type="mesh" mesh="tetra" contype="0" conaffinity="0"/>
          <body name="tip" pos="0 0.1 0"><joint name="d" axis="1 0 0"/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body>
        </body>
      </body></worldbody>
    </mujoco>""")
    assets = mjcf.load(path)
    assert assets.actuated_joint_names == ["a", "b", "c", "d"]
    np.testing.assert_allclose(
        assets.visual.vertices @ MUJOCO_TO_MODEL, [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]
    )
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    pose = np.array([0.4, -0.7, 1.1, 0.5])
    data.qpos[:] = pose
    mujoco.mj_kinematics(model, data)
    skeleton = rigid.forward_skeleton(
        assets, pose, global_rotation=None, global_translation=None, joint_indices=None, xp=np
    )
    np.testing.assert_allclose(skeleton[:, :3, 3], (data.xpos[1:] - data.xpos[1]) @ MUJOCO_TO_MODEL.T, atol=1e-6)
    rotations = MUJOCO_TO_MODEL @ data.xmat[1:].reshape(-1, 3, 3) @ MUJOCO_TO_MODEL.T
    np.testing.assert_allclose(skeleton[:, :3, :3], rotations, atol=1e-6)


@pytest.mark.fast
def test_a_driver_reaches_only_what_its_coupled_hinges_allow(tmp_path):
    path = tmp_path / "coupled.xml"
    path.write_text("""<mujoco>
      <compiler angle="radian"/>
      <worldbody><body name="root">
        <body name="driven"><joint name="driver" axis="1 0 0" range="-1 1"/>
          <body name="follower"><joint name="coupled" axis="1 0 0" range="0 1"/></body>
        </body>
        <body name="free"><joint name="unenforced" axis="1 0 0" range="-1 1" limited="false"/></body>
      </body></worldbody>
      <equality><joint joint1="coupled" joint2="driver" polycoef="0 2 0 0 0"/></equality>
    </mujoco>""")
    assets = mjcf.load(path)
    assert assets.actuated_joint_names == ["driver", "unenforced"]
    np.testing.assert_allclose(assets.actuated_joint_limits, [[0, 0.5], [-np.inf, np.inf]])
