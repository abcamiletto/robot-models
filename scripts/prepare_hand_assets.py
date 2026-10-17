"""Prepare canonical hands from pinned ROS/manufacturer-endorsed sources.

uv run --group asset-preparation python -m scripts.prepare_hand_assets UPSTREAMS OUTPUT

UPSTREAMS contains barrett/, robotiq/ and ezgripper-sim/ Git checkouts.
Original source files and licenses are retained unchanged. No upload is performed.
"""

import argparse
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh

from scripts import _hand_asset_tools as bundle

SOURCES = {
    "barrett": {
        "checkout": "barrett",
        "upstream": "https://github.com/RobotnikAutomation/barrett_hand_common",
        "revision": "b3d96febf8ef7b2aae4dab6015928918324c364a",
        "package": "barrett_hand_description",
        "description": "barrett_hand_description/robots/bh_alone.urdf.xacro",
        "license": "BSD-3-Clause",
    },
    "robotiq-3finger": {
        "checkout": "robotiq",
        "upstream": "https://github.com/ros-industrial-attic/robotiq",
        "revision": "45196f6558fe8ba9d89bc8a105396c68c3e7e892",
        "package": "robotiq_3f_gripper_visualization",
        "description": "robotiq_3f_gripper_visualization/cfg/robotiq-3f-gripper_articulated.xacro",
        "license": "BSD-2-Clause",
    },
    "ezgripper": {
        "checkout": "ezgripper-sim",
        "upstream": "https://github.com/vikashplus/ezgripper_sim",
        "revision": "0fdf9dbdb5023e9345f392a381cef80b1ebe965d",
        "description": "ezgripper.xml",
        "license": "Apache-2.0",
    },
}


def prepare(upstreams: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for name, spec in SOURCES.items():
        checkout = upstreams / spec["checkout"]
        bundle.check_revision(checkout, spec["revision"])
        target = output / name / "assets"
        source = target / "source"
        source.mkdir(parents=True)
        tracked = subprocess.check_output(["git", "-C", str(checkout), "ls-files", "-z"], text=True).split("\0")
        for filename in filter(None, tracked):
            path = Path(filename)
            keep = path.parts[0] == spec.get("package", "meshes") or filename in ("LICENSE", "README.md")
            if keep or filename == spec["description"]:
                destination = source / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(checkout / path, destination)
        if name == "ezgripper":
            mapping = prepare_mjcf(source / spec["description"], target)
        else:
            description = source / spec["description"]
            description = expand_xacro(source, spec, target / "expanded.urdf")
            mapping = convert_urdf(description, source, target)
        notice = (
            f"# {name} canonical assets\n\nSource: {spec['upstream']}, revision `{spec['revision']}`.\n\n"
            f"Upstream license: {spec['license']}.\n\n"
            "Original sources are unchanged under source/. Runtime XML uses a free root at the origin; "
            "display meshes are converted to STL. URDF rest transforms, joint axes/limits and mimic "
            "equalities are preserved. All URDF visuals are retained. Original collision geometry "
            "and dynamics remain available in the source description. The runtime is a kinematics/display "
            "model, not a simulation or motor-control replacement.\n\n" + (source / "LICENSE").read_text()
        )
        bundle.write_bundle(
            target,
            {
                "upstream": spec["upstream"],
                "revision": spec["revision"],
                "robot": name,
                "description": f"source/{spec['description']}",
                "mesh_dir": "source",
                "redistribution": "permitted under recorded upstream license",
                "licenses": [{"spdx_id": spec["license"], "text": "source/LICENSE"}],
                **mapping,
            },
            notice,
        )


def expand_xacro(source: Path, spec: dict, output: Path) -> Path:
    """Resolve ROS package lookups in a temporary copy; original xacros stay unchanged."""
    import xacro

    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary)
        shutil.copytree(source / spec["package"], stage / spec["package"])
        for path in stage.rglob("*.xacro"):
            text = path.read_text().replace(f"$(find {spec['package']})", str(stage / spec["package"]))
            path.write_text(text)
        document = xacro.process_file(str(stage / spec["description"]))
        output.write_text(document.toxml().replace(str(stage), "source"))
    return output


def source_mesh(source: Path, filename: str) -> trimesh.Trimesh:
    relative = filename.removeprefix("package://")
    path = source / relative
    mesh = trimesh.load_scene(path, process=False).to_geometry()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected triangular geometry: {path}")
    return mesh


def convert_urdf(description: Path, source: Path, output: Path) -> dict:
    """Preserve original URDF coordinates, all visuals and declared mimic relationships."""
    urdf = ET.parse(description).getroot()
    links = {link.attrib["name"]: link for link in urdf.findall("link")}
    joints = urdf.findall("joint")
    children = {bundle.required(joint, "child").attrib["link"] for joint in joints}
    (root_name,) = set(links) - children
    root, assets, world = bundle.runtime_xml()
    mimics = {}

    def visit(name: str, parent: ET.Element, joint: ET.Element | None = None) -> None:
        attrs = {} if joint is None else bundle.transform_attrs(bundle.origin(joint.find("origin")))
        body = ET.SubElement(parent, "body", {"name": name, **attrs})
        if joint is None:
            ET.SubElement(body, "freejoint")
        elif joint.get("type") != "fixed":
            if joint.get("type") != "revolute":
                raise ValueError(f"Unsupported URDF joint: {joint.attrib}")
            limit = bundle.required(joint, "limit").attrib
            axis = joint.find("axis")
            ET.SubElement(
                body,
                "joint",
                name=joint.attrib["name"],
                axis=axis.get("xyz", "1 0 0") if axis is not None else "1 0 0",
                range=f"{limit['lower']} {limit['upper']}",
            )
            mimic = joint.find("mimic")
            if mimic is not None:
                mimics[joint.attrib["name"]] = dict(mimic.attrib)
        for index, visual in enumerate(links[name].findall("visual")):
            geometry = bundle.required(visual, "geometry")
            mesh_tag = geometry.find("mesh")
            if mesh_tag is not None:
                mesh = source_mesh(source, mesh_tag.attrib["filename"])
                mesh.vertices *= bundle.vec(mesh_tag.get("scale"), "1 1 1")
            elif geometry.find("box") is not None:
                mesh = trimesh.creation.box(extents=bundle.vec(bundle.required(geometry, "box").get("size")))
            else:
                raise ValueError(f"Unsupported visual: {name}")
            mesh.apply_transform(bundle.origin(visual.find("origin")))
            bundle.add_mesh(mesh, f"{name}_visual_{index}", body, assets, output)
        for child in joints:
            if bundle.required(child, "parent").get("link") == name:
                visit(bundle.required(child, "child").attrib["link"], body, child)

    visit(root_name, world)
    if mimics:
        equalities = ET.SubElement(root, "equality")
        for name, mimic in mimics.items():
            ET.SubElement(
                equalities,
                "joint",
                joint1=name,
                joint2=mimic["joint"],
                polycoef=f"{mimic.get('offset', '0')} {mimic.get('multiplier', '1')} 0 0 0",
            )
    bundle.write_xml(root, output / "right.xml")
    independent = [joint for joint in world.findall(".//joint") if joint.attrib["name"] not in mimics]
    return {
        "root_link": root_name,
        "source_mimic": mimics,
        "side": "right",
        "joint_names": [joint.attrib["name"] for joint in independent],
        "joint_limits": [bundle.vec(joint.attrib["range"]).tolist() for joint in independent],
        "coupling_policy": "declared URDF mimic equalities",
        **bundle.RUNTIME_METADATA,
    }


def prepare_mjcf(description: Path, output: Path) -> dict:
    """Retain the endorsed MJCF tree and physical definitions; make a floating display root."""
    root = ET.parse(description).getroot()
    world = bundle.required(root, "worldbody")
    (body,) = world.findall("body")
    for element in list(world):
        if element is not body:
            world.remove(element)
    body.attrib.pop("pos", None)
    body.attrib.pop("quat", None)
    ET.SubElement(body, "freejoint")
    assets = bundle.required(root, "asset")
    for mesh in assets.findall("mesh"):
        path = description.parent / mesh.attrib["file"]
        destination = output / "meshes" / path.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, destination)
        mesh.set("file", f"meshes/{path.name}")
    bundle.write_xml(root, output / "right.xml")
    joints = body.findall(".//joint")
    return {
        "root_link": body.attrib["name"],
        "side": "right",
        "joint_names": [joint.attrib["name"] for joint in joints],
        "joint_limits": [bundle.vec(joint.attrib["range"]).tolist() for joint in joints],
        "coupling_policy": "independent hinge poses; native tendon dynamics retained but not solved by FK",
        **bundle.RUNTIME_METADATA,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstreams", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.upstreams, args.output)
