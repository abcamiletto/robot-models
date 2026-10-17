"""Prepare baseline source bundles and derived runtime MJCFs (no network or upload).

uv run python scripts/prepare_baseline_assets.py --gendexgrasp CHECKOUT \
    --gendex-archive data.zip --dexgraspnet CHECKOUT --output OUTPUT

The output follows <identifier>/{assets.zip,LICENSE.md} hub conventions and also
contains extracted assets/ directories for local use. Source bytes are unchanged.
Unresolved GenDexGrasp asset licenses prevent claiming these are publishable.
"""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import trimesh

GENDEX_REVISION = "29fe7efc558b8cc3ce2d8e67c9c22b014adc21c7"
DEX_REVISION = "bd1b13d7248729af117e1d46aaa6266b147a3c7b"
ARCHIVE_SHA256 = "c8777d2195da312281f51c37cdadd8a46bb1b408aada2e9ffd425bd237357f4e"
GENDEX_URL = "https://github.com/tengyu-liu/GenDexGrasp"
DEX_URL = "https://github.com/PKU-EPIC/DexGraspNet"
RUNTIME_METADATA = {
    "runtime_model": True,
    "source_to_model_basis": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    "pose_mapping": "identity in joint_names order: same angles and signs",
    "runtime_scope": "kinematics/display only; compiler mass/inertia bounds are placeholders, not source dynamics",
}

ROBOTS = {
    "barrett": "barrett-gendexgrasp",
    "robotiq_3finger": "robotiq-3finger-gendexgrasp",
    "ezgripper": "ezgripper-gendexgrasp",
    "shadowhand": "shadow-gendexgrasp",
    "allegro": "allegro-gendexgrasp",  # Source-only; reuse the existing left Allegro model.
}


def prepare(gendex: Path, archive_path: Path, dex: Path, output: Path) -> None:
    check_revision(gendex, GENDEX_REVISION)
    check_revision(dex, DEX_REVISION)
    if sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError("GenDexGrasp data.zip hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    prepare_gendex(archive_path, output)
    prepare_dexgraspnet(dex, output)


def prepare_gendex(archive_path: Path, output: Path) -> None:
    metadata_path = "data/urdf/urdf_assets_meta.json"
    with zipfile.ZipFile(archive_path) as archive:
        metadata = json.loads(archive.read(metadata_path))
        for robot, identifier in ROBOTS.items():
            target = output / identifier / "assets"
            source = target / "source"
            relative = Path(metadata["urdf_path"][robot])
            directory = f"data/urdf/{relative.parts[2]}/"
            annotations = f"data/urdf/contact_{robot}.json"
            files = [
                name
                for name in archive.namelist()
                if name.startswith(directory) or name in (metadata_path, annotations)
            ]
            archive.extractall(source, members=files)
            if robot != "allegro":
                mapping = convert_urdf(robot, source / relative, source / metadata["meshes_path"][robot], target)
            else:
                mapping = {"reuse_model": "allegro", "side": "left", "runtime_model": False}
            license_text = (
                f"# {identifier} source assets\n\nSource: {GENDEX_URL}, revision `{GENDEX_REVISION}`.\n\n"
                f"Original data.zip SHA-256: `{ARCHIVE_SHA256}`.\n\n"
                "Original descriptions, meshes, package metadata and annotations are retained byte-for-byte. "
                "Runtime MJCF and display STL files are derived by prepare_baseline_assets.py; see source.json.\n\n"
                "Redistribution status: unresolved. The archive does not include full asset license texts. "
                "EZGripper package.xml declares BSD; Shadow package.xml declares Private. "
                "Barrett, Robotiq and Allegro have no license file in their supplied directories. "
                "GenDexGrasp's public download is not treated as a blanket redistribution grant. "
                "Keep local until the applicable asset permissions are established.\n"
            )
            write_bundle(
                target,
                {
                    "upstream": GENDEX_URL,
                    "revision": GENDEX_REVISION,
                    "robot": robot,
                    "archive_sha256": ARCHIVE_SHA256,
                    "description": f"source/{relative}",
                    "mesh_dir": f"source/{metadata['meshes_path'][robot]}",
                    "annotations": {"contacts": f"source/data/urdf/contact_{robot}.json"},
                    **mapping,
                },
                license_text,
            )


def prepare_dexgraspnet(dex: Path, output: Path) -> None:
    target = output / "shadow-dexgraspnet" / "assets"
    source = target / "source"
    source.mkdir(parents=True)
    shutil.copytree(dex / "grasp_generation/mjcf", source / "mjcf")
    notice = dex / "grasp_generation/open_ai_assets/LICENSE.md"
    shutil.copyfile(notice, source / "OPENAI_ASSETS_LICENSE.md")
    # Preserve the pinned project's own license statement in addition to the inherited asset notice.
    shutil.copyfile(dex / "README.md", source / "UPSTREAM_README.md")
    mapping = convert_mjcf(source / "mjcf/shadow_hand_wrist_free.xml", target)
    write_bundle(
        target,
        {
            "upstream": DEX_URL,
            "revision": DEX_REVISION,
            "description": "source/mjcf/shadow_hand_wrist_free.xml",
            "mesh_dir": "source/mjcf/meshes",
            "annotations": {
                "contacts": "source/mjcf/contact_points.json",
                "penetration": "source/mjcf/penetration_points.json",
            },
            **mapping,
        },
        f"# DexGraspNet Shadow assets\n\nSource: {DEX_URL}, revision `{DEX_REVISION}`.\n\n"
        "DexGraspNet declares CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/ . "
        "Attribution: Ruicheng Wang, Jialiang Zhang, Jiayi Chen, Yinzhen Xu, Puhao Li, Tengyu Liu, He Wang, "
        "DexGraspNet (2022). Retain the noncommercial restriction on project-specific assets/annotations.\n\n"
        "Source files are unchanged; runtime XML and display STLs are derived. "
        "Display capsules use explicit symmetric centering, independent of Trimesh version. "
        "Native analytic collision and annotations remain in source/.\n\n" + notice.read_text(),
    )


def convert_urdf(robot: str, description: Path, mesh_dir: Path, output: Path) -> dict:
    """Match GenDexGrasp HandModel: first visual, Shadow mesh Rx(pi/2), independent mimic hinges."""
    urdf = ET.parse(description).getroot()
    links = {link.attrib["name"]: link for link in urdf.findall("link")}
    joints = urdf.findall("joint")
    child_names = {required(joint, "child").attrib["link"] for joint in joints}
    (root_name,) = set(links) - child_names
    root, assets, world = runtime_xml()
    order, limits, mimics = [], [], {}

    def visit(name: str, parent: ET.Element, joint: ET.Element | None = None) -> None:
        attrs = {} if joint is None else transform_attrs(origin(joint.find("origin")))
        runtime_name = "ezgripper_world" if robot == "ezgripper" and name == "world" else name
        body = ET.SubElement(parent, "body", {"name": runtime_name, **attrs})
        if joint is None:
            ET.SubElement(body, "freejoint")
        elif joint.get("type") != "fixed":
            if joint.get("type") != "revolute":
                raise ValueError(f"Unsupported source joint: {joint.attrib}")
            limit = required(joint, "limit").attrib
            axis = joint.find("axis")
            ET.SubElement(
                body,
                "joint",
                name=joint.attrib["name"],
                type="hinge",
                axis=axis.get("xyz", "1 0 0") if axis is not None else "1 0 0",
                range=f"{limit['lower']} {limit['upper']}",
            )
            order.append(joint.attrib["name"])
            limits.append([float(limit["lower"]), float(limit["upper"])])
            if joint.find("mimic") is not None:
                mimics[joint.attrib["name"]] = dict(required(joint, "mimic").attrib)
        visual = links[name].find("visual")
        if visual is not None:
            geometry = required(visual, "geometry")
            source_mesh = geometry.find("mesh")
            if source_mesh is not None:
                filename = source_mesh.attrib["filename"]
                if robot in ("shadowhand", "barrett"):
                    filename = Path(filename).name
                mesh = trimesh.load_mesh(mesh_dir / filename, process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError(f"Expected a triangular mesh: {filename}")
                mesh.vertices *= vec(source_mesh.get("scale"), "1 1 1")
            elif geometry.find("box") is not None:
                mesh = trimesh.creation.box(extents=vec(required(geometry, "box").get("size")))
            elif geometry.find("cylinder") is not None:
                cylinder = required(geometry, "cylinder")
                radius = float(cylinder.attrib["radius"])
                height = float(cylinder.attrib["length"])
                mesh = trimesh.creation.cylinder(radius=radius, height=height)
            else:
                raise ValueError(f"Unsupported source visual on {name}")
            if robot == "shadowhand":
                # This is in HandModel.py, not in the URDF. Apply before visual origin.
                mesh.vertices = mesh.vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
            mesh.apply_transform(origin(visual.find("origin")))
            add_mesh(mesh, name, body, assets, output)
        for child in joints:
            if required(child, "parent").get("link") == name:
                visit(required(child, "child").attrib["link"], body, child)

    visit(root_name, world)
    side = "left" if robot == "ezgripper" else "right"
    write_xml(root, output / f"{side}.xml")
    return {
        **RUNTIME_METADATA,
        "joint_names": order,
        "joint_limits": limits,
        "source_mimic": mimics,
        "coupling_policy": "independent, matching vendored pytorch_kinematics; URDF mimic retained in source",
        "side": side,
        "root_link": root_name,
        "source_link_to_model": {"world": "ezgripper_world"} if robot == "ezgripper" else {},
        "mesh_policy": "first visual; scale, then Shadow Rx(pi/2), then visual origin",
    }


def convert_mjcf(description: Path, output: Path) -> dict:
    """Preserve native body and hinge frames; tessellate collision primitives only for display."""
    source = ET.parse(description).getroot()
    root, assets, world = runtime_xml()
    order, limits = [], []

    def visit(original: ET.Element, parent: ET.Element, is_root: bool = False) -> None:
        attrs = {key: value for key, value in original.attrib.items() if key in ("name", "pos", "quat")}
        anchor = ET.SubElement(parent, "body", attrs)
        if is_root:
            ET.SubElement(anchor, "freejoint")
        body = anchor
        joints = original.findall("joint")
        if len(joints) > 1:
            raise ValueError("Expected at most one native joint per body")
        if joints:
            # The vendored MJCF parser exposes both a fixed pre-joint body and
            # a moving <name>_child link. Annotation keys address the latter.
            body = ET.SubElement(anchor, "body", name=original.attrib["name"] + "_child")
            joint = copy.deepcopy(joints[0])
            joint.set("type", "hinge")
            body.append(joint)
            order.append(joint.attrib["name"])
            limits.append(vec(joint.get("range")).tolist())
        parts = []
        for geom in original.findall("geom"):
            size = vec(geom.get("size"), "")
            if geom.get("type") == "box":
                # HandModel uses its supplied bevelled box.obj, not an ideal box.
                mesh = trimesh.load_mesh(description.parent / "meshes/box.obj", process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError("Expected triangular box.obj")
                mesh.vertices *= size
            elif geom.get("type") == "capsule":
                mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=(32, 32))
                mesh.vertices[:, 2] -= mesh.bounds[:, 2].mean()
            else:
                raise ValueError(f"Unexpected original collision geom: {geom.attrib}")
            transform = np.eye(4)
            if geom.get("quat"):
                transform = trimesh.transformations.quaternion_matrix(vec(geom.get("quat")))
            transform[:3, 3] = vec(geom.get("pos"))
            mesh.apply_transform(transform)
            parts.append(mesh)
        if parts:
            add_mesh(trimesh.util.concatenate(parts), body.attrib["name"], body, assets, output)
        for child in original.findall("body"):
            visit(child, body)

    visit(required(source, "worldbody/body"), world, is_root=True)
    write_xml(root, output / "right.xml")
    return {
        **RUNTIME_METADATA,
        "joint_names": order,
        "joint_limits": limits,
        "side": "right",
        "root_link": "robot0:palm",
        "coupling_policy": "22 independent hinges, no canonical wrist coordinates",
        "mesh_policy": "original bevelled box.obj and tessellated analytic capsules for display; use original MJCF for collision",
    }


def write_bundle(output: Path, metadata: dict, license_text: str) -> None:
    source_hashes = {
        str(path.relative_to(output)): sha256(path) for path in sorted((output / "source").rglob("*")) if path.is_file()
    }
    manifest = {"format_version": 1, **metadata, "source_sha256": source_hashes}
    (output / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "LICENSE.md").write_text(license_text)
    (output.parent / "LICENSE.md").write_text(license_text)
    with zipfile.ZipFile(output.parent / "assets.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.rglob("*")):
            if file.is_file():
                info = zipfile.ZipInfo(str(file.relative_to(output)), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, file.read_bytes())


def runtime_xml() -> tuple[ET.Element, ET.Element, ET.Element]:
    root = ET.Element("mujoco")
    root.append(ET.Comment("Derived by prepare_baseline_assets.py; original files are under source/."))
    ET.SubElement(
        root,
        "compiler",
        angle="radian",
        meshdir="meshes",
        discardvisual="false",
        boundmass="0.001",
        boundinertia="0.000001",
    )
    return root, ET.SubElement(root, "asset"), ET.SubElement(root, "worldbody")


def add_mesh(mesh: trimesh.Trimesh, name: str, body: ET.Element, assets: ET.Element, output: Path) -> None:
    filename = f"{name.replace(':', '_')}.stl"
    (output / "meshes").mkdir(exist_ok=True)
    mesh.export(output / "meshes" / filename)
    ET.SubElement(assets, "mesh", name=name, file=filename)
    ET.SubElement(body, "geom", type="mesh", mesh=name, contype="0", conaffinity="0", density="1000")


def write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)


def check_revision(path: Path, revision: str) -> None:
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise ValueError(f"Expected {revision} in {path}, got {actual}")
    subprocess.run(["git", "-C", str(path), "diff", "--quiet", "HEAD", "--", "."], check=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required(parent: ET.Element, path: str) -> ET.Element:
    child = parent.find(path)
    if child is None:
        raise ValueError(f"Missing {path} in {parent.tag}")
    return child


def origin(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.eye(4)
    transform = trimesh.transformations.euler_matrix(*vec(element.get("rpy")), axes="sxyz")
    transform[:3, 3] = vec(element.get("xyz"))
    return transform


def transform_attrs(transform: np.ndarray) -> dict[str, str]:
    return {
        "pos": numbers(transform[:3, 3]),
        "quat": numbers(trimesh.transformations.quaternion_from_matrix(transform)),
    }


def numbers(value: np.ndarray) -> str:
    return " ".join(format(float(x), ".17g") for x in value)


def vec(value: str | None, default: str = "0 0 0") -> np.ndarray:
    return np.fromstring(value if value is not None else default, sep=" ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gendexgrasp", type=Path, required=True)
    parser.add_argument("--gendex-archive", type=Path, required=True)
    parser.add_argument("--dexgraspnet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.gendexgrasp, args.gendex_archive, args.dexgraspnet, args.output)
