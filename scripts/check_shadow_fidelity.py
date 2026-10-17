"""Compare packaged Shadow geometry and kinematics with the original Menagerie MJCF.

uv run --group asset-preparation python -m scripts.check_shadow_fidelity MENAGERIE ASSETS OUTPUT
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts import _hand_asset_tools as bundle
from scripts import prepare_shadow_assets as preparation


def check(checkout: Path, assets: Path, output: Path) -> None:
    bundle.check_revision(checkout, preparation.REVISION)
    reports = []
    directions = np.vstack((np.eye(3), -np.eye(3), np.random.default_rng(4).normal(size=(48, 3))))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for side in ("left", "right"):
        source = mujoco.MjModel.from_xml_path(str(checkout / "shadow_hand" / f"{side}_hand.xml"))
        runtime = mujoco.MjModel.from_xml_path(str(assets / f"{side}.xml"))
        assert [source.body(i).name for i in range(source.nbody)] == [
            runtime.body(i).name for i in range(runtime.nbody)
        ]
        assert [source.joint(i).name for i in range(source.njnt)] == [
            runtime.joint(i).name for i in range(runtime.njnt)
        ]
        for attr in (
            "body_parentid",
            "jnt_type",
            "jnt_bodyid",
            "jnt_axis",
            "jnt_pos",
            "jnt_range",
            "body_mass",
            "body_inertia",
        ):
            np.testing.assert_allclose(getattr(runtime, attr), getattr(source, attr), atol=1e-10, err_msg=attr)
        for attr in ("geom_type", "geom_bodyid", "geom_contype", "geom_conaffinity"):
            np.testing.assert_array_equal(getattr(runtime, attr), getattr(source, attr), err_msg=attr)
        primitives = source.geom_type != mujoco.mjtGeom.mjGEOM_MESH
        np.testing.assert_allclose(runtime.geom_size[primitives], source.geom_size[primitives], atol=1e-12)
        lo, hi = source.jnt_range.T
        poses = [np.clip(np.zeros(source.nq), lo, hi), lo, hi, (lo + hi) / 2, lo * 0.63 + hi * 0.37]
        source_data, runtime_data = mujoco.MjData(source), mujoco.MjData(runtime)
        body_error, surface_error = 0.0, 0.0
        for pose in poses:
            source_data.qpos[:] = pose
            runtime_data.qpos[:] = pose
            mujoco.mj_kinematics(source, source_data)
            mujoco.mj_kinematics(runtime, runtime_data)
            for attr in ("xpos", "xmat"):
                a, b = getattr(runtime_data, attr), getattr(source_data, attr)
                body_error = max(body_error, float(np.abs(a - b).max()))
                np.testing.assert_allclose(a, b, atol=1e-10)
            for geom in range(source.ngeom):
                if source.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                    np.testing.assert_allclose(runtime_data.geom_xpos[geom], source_data.geom_xpos[geom], atol=1e-10)
                    np.testing.assert_allclose(runtime_data.geom_xmat[geom], source_data.geom_xmat[geom], atol=1e-10)
                    continue
                supports = []
                for model, data in ((source, source_data), (runtime, runtime_data)):
                    mesh = model.geom_dataid[geom]
                    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                    vertices = model.mesh_vert[start : start + count]
                    world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                    supports.append((world @ directions.T).max(axis=0))
                error = float(np.abs(supports[0] - supports[1]).max())
                surface_error = max(surface_error, error)
                np.testing.assert_allclose(supports[0], supports[1], atol=2e-6)
        collisions = np.count_nonzero(source.geom_contype | source.geom_conaffinity)
        report = {
            "side": side,
            "source_revision": preparation.REVISION,
            "poses": len(poses),
            "collisions": int(collisions),
            "body_max_abs": body_error,
            "mesh_support_max_m": surface_error,
        }
        reports.append(report)
        print(json.dumps(report))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    check(args.checkout, args.assets, args.output)
