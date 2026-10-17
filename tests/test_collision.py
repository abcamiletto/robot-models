"""Collision selection and primitive frames against MuJoCo's distance queries."""

import numpy as np
import pytest
import trimesh

from robot_models._common import mjcf


@pytest.mark.fast
def test_collision_defaults_frames_and_explicit_pairs(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    trimesh.creation.box(extents=[0.1, 0.08, 0.06]).export(tmp_path / "box.stl")
    path = tmp_path / "hand.xml"
    path.write_text("""<mujoco>
      <compiler angle="radian"/>
      <default><default class="collision"><geom contype="2" conaffinity="4"/>
        <default class="capsule"><geom type="capsule" size="0.03"/></default>
      </default></default>
      <asset><mesh name="box" file="box.stl"/></asset>
      <worldbody><body name="hand">
        <geom name="visual" type="mesh" mesh="box" contype="0" conaffinity="0"/>
        <geom name="capsule" class="capsule" fromto="0.1 0.2 0.3 0.1 0.2 0.1"/>
        <geom name="sphere" class="collision" type="sphere" size="0.04" pos="0.2 0 0"/>
        <geom name="box" class="collision" type="box" size="0.03 0.04 0.05" xyaxes="0 1 0 0 0 1"/>
        <geom name="cylinder" class="collision" type="cylinder" size="0.03 0.06"
              axisangle="1 2 3 0.7" pos="0 0.2 0"/>
        <geom name="ellipsoid" class="collision" type="ellipsoid" size="0.03 0.05 0.07"
              zaxis="1 2 3" pos="0 0 0.2"/>
        <geom name="paired" type="mesh" mesh="box" pos="0.2 0.2 0.2" contype="0" conaffinity="0"/>
      </body></worldbody>
      <contact><pair geom1="paired" geom2="sphere"/></contact>
    </mujoco>""")
    assets = mjcf.load(path)
    assert assets.collision.names == ["capsule", "sphere", "box", "cylinder", "ellipsoid", "paired"]
    # A point-sized sphere independently probes the source collision surfaces.
    spec = mujoco.MjSpec.from_file(str(path))
    probe = spec.worldbody.add_body(name="probe", mocap=True)
    probe.add_geom(name="probe", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[1e-7, 0, 0])
    model = spec.compile()
    data = mujoco.MjData(model)
    for index, name in enumerate(assets.collision.names):
        start, end = assets.collision.vertex_offsets[index : index + 2]
        native = assets.collision.vertices[start:end] @ mjcf.MUJOCO_TO_MODEL
        for point in native[np.linspace(0, len(native) - 1, 12, dtype=int)]:
            data.mocap_pos[0] = point
            mujoco.mj_kinematics(model, data)
            distance = mujoco.mj_geomDistance(model, data, model.geom(name).id, model.geom("probe").id, 1.0, None)
            np.testing.assert_allclose(distance, -1e-7, atol=2e-6, err_msg=name)
