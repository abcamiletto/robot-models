"""Validate prepared hands against the actual pinned baseline code in separate processes.

Use the fully provisioned audit interpreter with PYTHONPATH=src, or an audit uv
project with its declared upstream dependencies. Never import both vendored
pytorch_kinematics packages in the same process. No baseline sources are written.
See docs/models/baselines.md for commands and fixture provenance.
"""

import argparse
import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy import spatial

# Ensure this checkout wins over any old installed robot-models in the audit interpreter.
sys.dont_write_bytecode = True  # Keep read-only baseline trees free of import caches.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import robot_models
from robot_models import baselines
from robot_models._common.mjcf import MUJOCO_TO_MODEL as C

GENDEX_REVISION = "29fe7efc558b8cc3ce2d8e67c9c22b014adc21c7"
DEX_REVISION = "bd1b13d7248729af117e1d46aaa6266b147a3c7b"
NAMES = {
    "barrett": "barrett-gendexgrasp",
    "robotiq_3finger": "robotiq-3finger-gendexgrasp",
    "ezgripper": "ezgripper-gendexgrasp",
    "shadowhand": "shadow-gendexgrasp",
}


def check_gendex(args):
    check_revision(args.upstream, GENDEX_REVISION)
    reports = []
    for robot, identifier in NAMES.items():
        original, metadata = gendex_hand(args.upstream, args.original_assets, robot)
        family = "shadow" if robot == "shadowhand" else identifier
        options = {"source": "gendexgrasp"} if robot == "shadowhand" else {}
        bundle = baselines.source_assets(family, model_path=args.bundles / identifier / "assets", **options)
        bundle.verify()
        # Compare actual external source assets, not only preparation-time hashes.
        assert bundle.description.read_bytes() == (args.original_assets / metadata["urdf_path"][robot]).read_bytes()
        original_contacts = args.original_assets / f"data/urdf/contact_{robot}.json"
        assert bundle.annotations["contacts"].read_bytes() == original_contacts.read_bytes()
        model = robot_models.create_model(family, model_path=args.bundles / identifier / "assets", **options)
        names = original.robot.get_joint_parameter_names()
        urdf = ET.parse(bundle.description).getroot()
        joint_by_name = {joint.attrib["name"]: joint for joint in urdf.findall("joint")}
        limits = np.asarray(
            [[float(joint_by_name[name].find("limit").get(key)) for key in ("lower", "upper")] for name in names]
        )
        # Original hand annotations are vertex-index bases; resolve using original HandModel, no resampling.
        points = {name: value[0, :, :3].numpy() for name, value in original.contact_point_basis.items()}
        meshes = [(name, vertices, original.mesh_faces[name]) for name, vertices in original.mesh_verts.items()]
        link_mapping = bundle.manifest["source_link_to_model"]
        annotations = {"contact_basis": points}
        report, arrays = compare_model(model, original.robot, names, limits, meshes, annotations, link_mapping)
        write_comparison(args.output, args.write_fixtures, report, arrays)
        reports.append(report)
    return reports


def check_dex(args):
    check_revision(args.upstream, DEX_REVISION)
    original = dex_hand(args.upstream)
    identifier = "shadow-dexgraspnet"
    bundle = baselines.source_assets("shadow", source="dexgraspnet", model_path=args.bundles / identifier / "assets")
    bundle.verify()
    native = args.upstream / "grasp_generation/mjcf"
    assert bundle.description.read_bytes() == (native / "shadow_hand_wrist_free.xml").read_bytes()
    annotations = {}
    for key, filename in (("contacts", "contact_points.json"), ("penetration", "penetration_points.json")):
        assert bundle.annotations[key].read_bytes() == (native / filename).read_bytes()
        annotations[key] = json.loads((native / filename).read_text())
    assert original.n_contact_candidates == 140
    assert original.n_keypoints == 19
    model = robot_models.create_model("shadow", source="dexgraspnet", model_path=args.bundles / identifier / "assets")
    limits = np.stack([original.joints_lower.numpy(), original.joints_upper.numpy()], axis=1)
    # Validate boxes against original mesh construction. Capsules use exact analytic surfaces below:
    # modern Trimesh changed the historical display capsule origin in original HandModel.
    meshes = [
        (name, item["vertices"].numpy(), item["faces"].numpy())
        for name, item in original.mesh.items()
        if "geom_param" not in item or len(item["geom_param"]) == 3
    ]
    names = original.chain.get_joint_parameter_names()
    report, arrays = compare_model(model, original.chain, names, limits, meshes, annotations, {})
    capsule_error = 0.0
    capsule_names, capsule_radii, capsule_heights = [], [], []
    for name, item in original.mesh.items():
        if "geom_param" not in item or len(item["geom_param"]) != 2:
            continue
        height = float(item["geom_param"][1]) * 2
        radius = float(item["geom_param"][0])
        capsule_names.append(name)
        capsule_radii.append(radius)
        capsule_heights.append(height)
        vertices = np.asarray(model.link_meshes[model.link_names.index(name)].vertices) @ C
        nearest = np.zeros_like(vertices)
        nearest[:, 2] = np.clip(vertices[:, 2], 0, height)
        error = np.abs(np.linalg.norm(vertices - nearest, axis=1) - radius).max()
        capsule_error = max(capsule_error, float(error))
        assert error < 2e-7, (name, error)
    arrays.update(
        capsule_names=np.asarray(capsule_names),
        capsule_radii=np.asarray(capsule_radii),
        capsule_heights=np.asarray(capsule_heights),
    )
    report.update(
        analytic_capsule_surface_max_m=capsule_error,
        capsules_checked=len(capsule_names),
        contacts=140,
        penetration_points=19,
    )
    write_comparison(args.output, args.write_fixtures, report, arrays)
    return [report]


def check_allegro(args):
    check_revision(args.upstream, GENDEX_REVISION)
    original, _ = gendex_hand(args.upstream, args.original_assets, "allegro")
    model = robot_models.create_model("allegro", side="left", model_path=args.allegro_assets)
    # Existing canonical Allegro has source-equivalent geometry in a different link/rest frame.
    source_names = original.robot.get_joint_parameter_names()
    source_indices = [int(name.removeprefix("joint_").split(".")[0]) for name in source_names]
    target_names = [f"{('ff', 'mf', 'rf', 'th')[i // 4]}j{i % 4}" for i in source_indices]
    source_to_target = [model.actuated_joint_names.index(name) for name in target_names]
    description = args.original_assets / "data/urdf/allegro_hand_description/allegro_hand_description_left.urdf"
    urdf = ET.parse(description).getroot()
    by_name = {joint.get("name"): joint for joint in urdf.findall("joint")}
    limits = np.array(
        [[float(by_name[name].find("limit").get(key)) for key in ("lower", "upper")] for name in source_names]
    )
    target_limits = np.asarray(model.actuated_joint_limits)[source_to_target]
    np.testing.assert_allclose(target_limits, limits, atol=1e-7)
    poses = fixed_poses(limits)
    candidate = np.zeros_like(poses)
    candidate[:, source_to_target] = poses
    status = original.robot.forward_kinematics(torch.from_numpy(poses))
    actual = model.forward_skeleton(candidate)
    # Map all articulated bodies by mechanical joint, with a fixed per-body rest transform.
    mappings = []
    max_error = 0.0
    source_palm = in_model_frame(status["base_link"].get_matrix().numpy())
    target_palm = actual[:, model.joint_names.index("palm")]
    for index, source_name in enumerate(source_names):
        source_body = by_name[source_name].find("child").get("link")
        target_body = model._assets.hinge_joint_indices[model.actuated_joint_names.index(target_names[index])]
        src = in_model_frame(status[source_body].get_matrix().numpy())
        # Align palm root first, then retain the source body's rest-frame orientation/offset.
        src = np.linalg.inv(source_palm) @ src
        dst = np.linalg.inv(target_palm) @ actual[:, target_body]
        rest = np.linalg.inv(dst[0]) @ src[0]
        error = np.abs(dst @ rest - src).max()
        max_error = max(max_error, float(error))
        assert error < 3e-6, (source_body, model.joint_names[target_body], error)
        mappings.append(
            {
                "source_joint": source_name,
                "target_joint": target_names[index],
                "sign": 1,
                "source_body": source_body,
                "target_body": model.joint_names[target_body],
                "target_to_source_rest": rest.tolist(),
            }
        )
    # Compare full original/canonical surfaces, palm-relative, at rest and two articulated fixed poses.
    # Sampling noise is bounded by also comparing independent samples of the original surface to itself.
    surface = []
    for row in (0, 4, 5):
        q = torch.zeros(1, 25)
        q[:, 3] = q[:, 7] = 1
        q[:, 9:] = torch.from_numpy(poses[row])
        source_mesh = trimesh.util.concatenate(original.get_meshes_from_q(q))
        source_mesh.apply_transform(np.linalg.inv(original.current_status["base_link"].get_matrix()[0].numpy()))
        source_mesh.vertices = source_mesh.vertices @ C.T
        target_mesh = model.forward_meshes(hand_pose=candidate[row])[0]
        target_mesh.apply_transform(np.linalg.inv(actual[row, model.joint_names.index("palm")]))
        source_points, _ = trimesh.sample.sample_surface(source_mesh, 30000, seed=42)
        control, _ = trimesh.sample.sample_surface(source_mesh, 30000, seed=17)
        target_points, _ = trimesh.sample.sample_surface(target_mesh, 30000, seed=17)
        diff = spatial.cKDTree(source_points).query(target_points)[0]
        noise = spatial.cKDTree(source_points).query(control)[0]
        surface.append(
            {
                "pose": row,
                "mean_m": float(diff.mean()),
                "p95_m": float(np.quantile(diff, 0.95)),
                "control_mean_m": float(noise.mean()),
                "control_p95_m": float(np.quantile(noise, 0.95)),
            }
        )
        assert diff.mean() < noise.mean() * 1.25
        assert np.quantile(diff, 0.95) < np.quantile(noise, 0.95) * 1.25
    report = {
        "model": "allegro",
        "side": "left",
        "fk_max_abs": max_error,
        "joint_mapping": mappings,
        "source_limits": limits.tolist(),
        "target_limits_in_source_order": target_limits.tolist(),
        "surfaces": surface,
    }
    (args.output / "allegro-left.json").write_text(json.dumps(report, indent=2) + "\n")
    return [report]


def compare_model(model, chain, names, limits, source_meshes, annotations, link_mapping):
    assert model.actuated_joint_names == names, (model.actuated_joint_names, names)
    np.testing.assert_allclose(model.actuated_joint_limits, limits, atol=1e-7)
    poses = fixed_poses(limits)
    status = chain.forward_kinematics(torch.from_numpy(poses))
    status = {link_mapping.get(name, name): value for name, value in status.items()}
    assert set(model.joint_names) == set(status)
    frames = {name: np.broadcast_to(value.get_matrix().numpy(), (len(poses), 4, 4)) for name, value in status.items()}
    expected = in_model_frame(np.stack([frames[name] for name in model.joint_names], axis=1))
    actual = model.forward_skeleton(poses)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    fk_error = float(np.max(np.abs(actual - expected)))
    point_indices, point_bodies, point_weights, point_expected = [], [], [], []
    mesh_max = 0.0
    for name, vertices, faces in source_meshes:
        assert name in model.link_names
        link = model.link_names.index(name)
        mesh = model.link_meshes[link]
        local = vertices @ C.T
        distance, indices = spatial.cKDTree(mesh.vertices).query(local)
        mesh_max = max(mesh_max, float(distance.max()))
        assert distance.max() < 2e-7, (name, distance.max())
        # STL deduplication must preserve every source triangle's geometry.
        runtime_faces = {tuple(sorted(face)) for face in np.asarray(mesh.faces)}
        assert all(tuple(sorted(face)) in runtime_faces for face in indices[faces]), name
        selected = faces[np.linspace(0, len(faces) - 1, min(8, len(faces)), dtype=int)]
        for face in selected:
            weights = np.array([0.2, 0.3, 0.5])
            local_point = weights @ vertices[face]
            world = frames[name][:, :3, :3] @ local_point + frames[name][:, :3, 3]
            point_expected.append(world @ C.T)
            point_indices.append(indices[face] + model._assets.link_vertex_starts[link])
            point_bodies.append(model.link_joint_indices[link])
            point_weights.append(weights)
    arrays = {
        "poses": poses,
        "joint_names": np.asarray(model.joint_names),
        "pose_names": np.asarray(names),
        "limits": limits,
        "transforms": expected.astype(np.float32),
        "point_indices": np.asarray(point_indices),
        "point_bodies": np.asarray(point_bodies),
        "point_weights": np.asarray(point_weights),
        "point_expected": np.stack(point_expected, axis=1).astype(np.float32),
    }
    annotation_error = 0.0
    for label, points_by_body in annotations.items():
        native, bodies = [], []
        for body, points in points_by_body.items():
            points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
            native.extend(points)
            bodies.extend([model.joint_names.index(body)] * len(points))
        if not bodies:
            continue
        native = np.asarray(native)
        point_frames = np.stack([frames[model.joint_names[body]] for body in bodies], axis=1)
        expected_points = np.einsum("bpij,pj->bpi", point_frames[:, :, :3, :3], native) + point_frames[:, :, :3, 3]
        expected_points = expected_points @ C.T
        model_frames = actual[:, bodies]
        model_points = native @ C.T
        actual_points = np.einsum("bpij,pj->bpi", model_frames[:, :, :3, :3], model_points) + model_frames[:, :, :3, 3]
        np.testing.assert_allclose(actual_points, expected_points, atol=2e-6)
        annotation_error = max(annotation_error, float(np.max(np.abs(actual_points - expected_points))))
        arrays[f"{label}_local"] = native @ C.T
        arrays[f"{label}_bodies"] = np.asarray(bodies)
        arrays[f"{label}_expected"] = expected_points
    report = {
        "model": model._NAME,
        "poses": len(poses),
        "joint_names": names,
        "limits": limits.tolist(),
        "fk_max_abs": fk_error,
        "mesh_vertex_max_m": mesh_max,
        "annotation_max_m": annotation_error,
        "source_mesh_links_checked": len(source_meshes),
    }
    if model._NAME == "shadow":
        report["source"] = model.source
    return report, arrays


def write_comparison(output, fixture_dir, report, arrays):
    identifier = report["model"]
    if "source" in report:
        identifier += "-" + report["source"]
    (output / f"{identifier}.json").write_text(json.dumps(report, indent=2) + "\n")
    if fixture_dir is not None:
        fixture_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fixture_dir / f"{identifier}.npz", **arrays)


def gendex_hand(checkout, extracted, robot):
    import transforms3d

    sys.path.insert(0, str(checkout / "thirdparty/pytorch_kinematics"))
    import pytorch_kinematics as pk

    assert Path(pk.__file__).resolve().is_relative_to(checkout / "thirdparty/pytorch_kinematics"), (
        "Wrong kinematics vendor loaded"
    )
    from pytorch_kinematics.urdf_parser_py import urdf as parser

    rotation_path = checkout / "utils/rot6d.py"
    names = [node.name for node in ast.parse(rotation_path.read_text()).body if isinstance(node, ast.FunctionDef)]
    rotations = definitions(rotation_path, names, {"torch": torch, "np": np})
    namespace = {
        "torch": torch,
        "np": np,
        "tm": trimesh,
        "trimesh": trimesh,
        "os": os,
        "json": json,
        "pk": pk,
        "URDF_PARSER": parser,
        "URDF": parser.URDF,
        "Mesh": parser.Mesh,
        "Box": parser.Box,
        "Cylinder": parser.Cylinder,
        "Sphere": parser.Sphere,
        "transforms3d": transforms3d,
        **rotations,
    }
    Hand = definitions(checkout / "utils_model/HandModel.py", ["HandModel"], namespace)["HandModel"]
    metadata = json.loads((extracted / "data/urdf/urdf_assets_meta.json").read_text())
    with working_directory(extracted), contextlib.redirect_stdout(io.StringIO()):
        hand = Hand(
            robot,
            str(extracted / metadata["urdf_path"][robot]),
            str(extracted / metadata["meshes_path"][robot]),
            batch_size=1,
            device="cpu",
            hand_scale=1.0,
        )
    return hand, metadata


def dex_hand(checkout):
    sys.path.insert(0, str(checkout / "thirdparty/pytorch_kinematics"))
    import pytorch_kinematics as pk

    assert Path(pk.__file__).resolve().is_relative_to(checkout / "thirdparty/pytorch_kinematics"), (
        "Wrong kinematics vendor loaded"
    )

    namespace = {
        "torch": torch,
        "np": np,
        "tm": trimesh,
        "os": os,
        "json": json,
        "pk": pk,
        "index_vertices_by_faces": lambda vertices, faces: vertices[faces],
    }
    # n_surface_points=0 avoids PyTorch3D sampling; the original chain and annotations are unchanged.
    Hand = definitions(checkout / "grasp_generation/utils/hand_model.py", ["HandModel"], namespace)["HandModel"]
    root = checkout / "grasp_generation/mjcf"
    with working_directory(root.parent):
        return Hand(
            str(root / "shadow_hand_wrist_free.xml"),
            str(root / "meshes"),
            str(root / "contact_points.json"),
            str(root / "penetration_points.json"),
            n_surface_points=0,
        )


def fixed_poses(limits):
    low, high = limits.T
    zero = np.clip(np.zeros(len(low)), low, high)
    poses = [np.zeros(len(low)), zero, low, high, 0.77 * low + 0.23 * high, 0.39 * low + 0.61 * high]
    for index in range(len(low)):
        pose = zero.copy()
        pose[index] += 0.071
        poses.append(pose)
    return np.asarray(poses, dtype=np.float32)


def in_model_frame(matrices):
    basis = np.eye(4)
    basis[:3, :3] = C
    return basis @ matrices @ basis.T


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError(f"Missing definitions in {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 - unchanged definitions from checked, pinned sources
    return namespace


def check_revision(path, revision):
    assert subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() == revision
    subprocess.run(["git", "-C", str(path), "diff", "--quiet", "HEAD", "--", "."], check=True)


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=("gendexgrasp", "dexgraspnet", "allegro"), required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--original-assets", type=Path)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--allegro-assets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-fixtures", type=Path)
    args = parser.parse_args()
    args.upstream = args.upstream.resolve()
    if args.original_assets:
        args.original_assets = args.original_assets.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    np.random.seed(17)
    torch.manual_seed(17)
    reports = {"gendexgrasp": check_gendex, "dexgraspnet": check_dex, "allegro": check_allegro}[args.project](args)
    print(json.dumps(reports, indent=2))
