"""Compare canonical hands with MuJoCo reading their upstream descriptions.

uv run --group asset-preparation python -m scripts.check_hand_fidelity BUNDLES OUTPUT

URDF geometry is removed only from the independent FK reference, preserving every
source link/joint. Visual surfaces are checked separately against original meshes.
EZGripper uses its untouched upstream MJCF, including native collision/tendon data.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from robot_models import create_model
from robot_models._common.mjcf import MUJOCO_TO_MODEL as C
from scripts import prepare_hand_assets as preparation


def check(bundles: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in preparation.SOURCES:
        directory = bundles / name / "assets"
        manifest = json.loads((directory / "source.json").read_text())
        for group in ("source_sha256", "license_sha256"):
            for relative, expected_hash in manifest[group].items():
                actual_hash = hashlib.sha256((directory / relative).read_bytes()).hexdigest()
                assert actual_hash == expected_hash, relative
        hand = create_model(name, model_path=directory)
        if name == "ezgripper":
            reference = mujoco.MjModel.from_xml_path(str((directory / manifest["description"]).resolve()))
            mimics = {}
        else:
            urdf = ET.parse(directory / "expanded.urdf").getroot()
            mimics = {
                joint.attrib["name"]: dict(joint.find("mimic").attrib)
                for joint in urdf.findall("joint")
                if joint.find("mimic") is not None
            }
            reference = urdf_reference(urdf)
        joint_names = [reference.joint(i).name for i in range(reference.njnt)]
        body_names = [reference.body(i).name for i in range(1, reference.nbody)]
        assert hand.joint_names == body_names
        pose_names = [joint for joint in joint_names if joint not in mimics]
        assert pose_names == hand.actuated_joint_names
        limits = np.array([reference.jnt_range[reference.joint(joint).id] for joint in pose_names])
        np.testing.assert_allclose(hand.actuated_joint_limits, limits, atol=1e-7)
        np.testing.assert_allclose(hand._assets.hinge_axes, reference.jnt_axis @ C.T, atol=1e-7)
        poses = fixed_poses(limits)
        data = mujoco.MjData(reference)
        frames = []
        for pose in poses:
            values = dict(zip(pose_names, pose, strict=True))

            for index in range(reference.njnt):
                data.qpos[reference.jnt_qposadr[index]] = angle(reference.joint(index).name, values, mimics)
            mujoco.mj_kinematics(reference, data)
            world = np.broadcast_to(np.eye(4), (len(body_names), 4, 4)).copy()
            world[:, :3, 3] = data.xpos[1:]
            world[:, :3, :3] = data.xmat[1:].reshape(-1, 3, 3)
            frames.append(np.linalg.inv(world[0]) @ world)
        frames = np.asarray(frames)
        basis = np.eye(4)
        basis[:3, :3] = C
        expected = basis @ frames @ basis.T
        actual = hand.forward_skeleton(poses)
        fk_error = float(np.max(np.abs(actual - expected)))
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        points = (
            mjcf_points(reference, data) if name == "ezgripper" else urdf_points(urdf, directory / "source", body_names)
        )
        visual_arrays, surface_error = surface_reference(hand, frames, points, "visual")
        collision_points = (
            mjcf_points(reference, data, collision=True)
            if name == "ezgripper"
            else urdf_points(urdf, directory / "source", body_names, element="collision")
        )
        collision_arrays, collision_error = surface_reference(hand, frames, collision_points, "collision")
        arrays = {
            "poses": poses,
            "joint_names": hand.joint_names,
            "pose_names": pose_names,
            "limits": limits,
            "transforms": expected.astype(np.float32),
            **visual_arrays,
            **{f"collision_{key}": value for key, value in collision_arrays.items()},
        }
        np.savez_compressed(output / f"{name}.npz", **arrays)
        report = {
            "model": name,
            "upstream": manifest["upstream"],
            "revision": manifest["revision"],
            "poses": len(poses),
            "pose_coordinates": hand.num_dofs,
            "hinges": reference.njnt,
            "mimic_joints": len(mimics),
            "visuals": len(points),
            "fk_max_abs": fk_error,
            "surface_max_m": surface_error,
            "collisions": len(collision_points),
            "collision_surface_max_m": collision_error,
        }
        (output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


def surface_reference(hand, frames: np.ndarray, points: list, geometry: str) -> tuple[dict, float]:
    """Bind original surface samples to runtime triangles, after checking conversion error."""
    triangles, weights, links, expected_points = [], [], [], []
    max_error = 0.0
    meshes = hand.link_meshes if geometry == "visual" else hand.collision_meshes
    assets = getattr(hand._assets, geometry)
    assert len(points) == len(meshes)
    for index, (body, local) in enumerate(points):
        mesh = meshes[index]
        # Millimetres avoid Trimesh's degeneracy tolerance on small CAD triangles.
        probe = trimesh.Trimesh(vertices=np.asarray(mesh.vertices) * 1000, faces=mesh.faces, process=False)
        closest, distance, triangle = probe.nearest.on_surface((local @ C.T) * 1000)
        max_error = max(max_error, float(distance.max() / 1000))
        assert distance.max() / 1000 < 2e-6, (geometry, body, distance.max() / 1000)
        weights.extend(trimesh.triangles.points_to_barycentric(probe.triangles[triangle], closest))
        triangles.extend(np.asarray(mesh.faces)[triangle] + assets.vertex_offsets[index])
        links.extend([index] * len(local))
        frame = frames[:, hand.joint_names.index(body)]
        expected_points.append((np.einsum("bij,pj->bpi", frame[:, :3, :3], local) + frame[:, None, :3, 3]) @ C.T)
    return {
        "point_triangles": triangles,
        "point_weights": weights,
        "point_links": links,
        "points": np.concatenate(expected_points, axis=1).astype(np.float32),
    }, max_error


def urdf_reference(original: ET.Element) -> mujoco.MjModel:
    root = ET.fromstring(ET.tostring(original))
    for link in root.findall("link"):
        for child in list(link):
            if child.tag in ("visual", "collision"):
                link.remove(child)
    compiler = ET.SubElement(ET.SubElement(root, "mujoco"), "compiler")
    compiler.attrib.update(fusestatic="false", boundmass="0.001", boundinertia="0.00001", balanceinertia="true")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def urdf_points(
    urdf: ET.Element, source: Path, bodies: list[str], *, element: str = "visual"
) -> list[tuple[str, np.ndarray]]:
    points = []
    # Match the source's tree order using its independent MuJoCo import.
    links = {link.attrib["name"]: link for link in urdf.findall("link")}
    for body in bodies:
        for visual in links[body].findall(element):
            geometry = visual.find("geometry")
            tag = geometry.find("mesh")
            if tag is None:
                mesh = trimesh.creation.box(extents=np.fromstring(geometry.find("box").attrib["size"], sep=" "))
            else:
                path = source / tag.attrib["filename"].removeprefix("package://")
                mesh = trimesh.load_scene(path, process=False).to_geometry()
                mesh.vertices *= np.fromstring(tag.get("scale", "1 1 1"), sep=" ")
            origin = visual.find("origin")
            position = np.zeros(3) if origin is None else np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.zeros(3) if origin is None else np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            rotation = Rotation.from_euler("xyz", rpy).as_matrix()
            points.append((body, sample(mesh.vertices, mesh.faces) @ rotation.T + position))
    return points


def mjcf_points(model: mujoco.MjModel, data: mujoco.MjData, *, collision: bool = False) -> list[tuple[str, np.ndarray]]:
    points = []
    paired = set(model.pair_geom1) | set(model.pair_geom2)
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] == 0:
            continue  # Source scene floors are not part of the hand.
        active = bool(model.geom_contype[geom] or model.geom_conaffinity[geom] or geom in paired)
        if active != collision:
            continue
        if not collision and model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            face_start, face_count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            local = sample(
                model.mesh_vert[start : start + count], model.mesh_face[face_start : face_start + face_count]
            )
        else:
            assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
            mesh = trimesh.creation.box(extents=2 * model.geom_size[geom])
            local = sample(mesh.vertices, mesh.faces)
        world = local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        body = model.geom_bodyid[geom]
        in_body = (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
        points.append((model.body(body).name, in_body))
    return points


def sample(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    selected = np.linspace(0, len(faces) - 1, min(16, len(faces)), dtype=int)
    triangles = vertices[faces[selected]]
    used = np.unique(faces)
    corners = vertices[used[np.linspace(0, len(used) - 1, min(16, len(used)), dtype=int)]]
    return np.concatenate([corners, np.einsum("i,pij->pj", [0.2, 0.3, 0.5], triangles)])


def fixed_poses(limits: np.ndarray) -> np.ndarray:
    middle = limits.mean(1)
    poses = [np.zeros(len(limits)), np.clip(np.zeros(len(limits)), *limits.T), middle]
    for coordinate in range(len(limits)):
        for endpoint in limits[coordinate]:
            pose = middle.copy()
            pose[coordinate] = endpoint
            poses.append(pose)
    poses.append(limits[:, 0] * 0.63 + limits[:, 1] * 0.37)
    return np.asarray(poses, dtype=np.float32)


def angle(joint: str, values: dict, mimics: dict) -> float:
    if joint not in mimics:
        return values[joint]
    mimic = mimics[joint]
    return float(mimic.get("offset", 0)) + float(mimic.get("multiplier", 1)) * angle(mimic["joint"], values, mimics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    check(args.bundles, args.output)
