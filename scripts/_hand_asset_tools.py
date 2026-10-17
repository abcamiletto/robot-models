"""Packaging helpers for canonical upstream hand assets."""

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import trimesh

RUNTIME_METADATA = {
    "runtime_model": True,
    "source_to_model_basis": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    "pose_mapping": "identity in joint_names order: same angles and signs",
    "runtime_scope": "kinematics and source collision geometry; original dynamics remain under source/",
}


def write_bundle(output: Path, metadata: dict, license_text: str) -> None:
    source_hashes = {
        str(path.relative_to(output)): sha256(path) for path in sorted((output / "source").rglob("*")) if path.is_file()
    }
    (output / "LICENSE.md").write_text(license_text)
    (output.parent / "LICENSE.md").write_text(license_text)
    license_files = [output / "LICENSE.md", *sorted((output / "licenses").rglob("*"))]
    license_hashes = {str(path.relative_to(output)): sha256(path) for path in license_files if path.is_file()}
    manifest = {"format_version": 1, **metadata, "source_sha256": source_hashes, "license_sha256": license_hashes}
    (output / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output.parent / "assets.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.rglob("*")):
            if file.is_file():
                info = zipfile.ZipInfo(str(file.relative_to(output)), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, file.read_bytes())


def runtime_xml() -> tuple[ET.Element, ET.Element, ET.Element]:
    root = ET.Element("mujoco")
    root.append(ET.Comment("Derived by prepare_hand_assets.py; original files are under source/."))
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


def add_mesh(
    mesh: trimesh.Trimesh, name: str, body: ET.Element, assets: ET.Element, output: Path, *, collision: bool = False
) -> None:
    filename = f"{name.replace(':', '_')}.stl"
    (output / "meshes").mkdir(exist_ok=True)
    mesh.export(output / "meshes" / filename)
    ET.SubElement(assets, "mesh", name=name, file=filename)
    attrs = (
        {"contype": "1", "conaffinity": "1", "density": "0", "group": "3"}
        if collision
        else {"contype": "0", "conaffinity": "0", "density": "1000"}
    )
    if collision:
        attrs["name"] = name
    ET.SubElement(body, "geom", {"type": "mesh", "mesh": name, **attrs})


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
