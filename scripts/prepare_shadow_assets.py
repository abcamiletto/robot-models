"""Restore Menagerie's forearm collision without changing existing visual meshes.

uv run --group asset-preparation python -m scripts.prepare_shadow_assets MENAGERIE BASE_ZIP OUTPUT
BASE_ZIP is shadow/assets.zip at the original robot-models hub snapshot below.
"""

import argparse
import copy
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import trimesh

from robot_models._common import mjcf
from scripts import _hand_asset_tools as bundle

UPSTREAM = "https://github.com/google-deepmind/mujoco_menagerie"
REVISION = "f054586a8e90465d49ee5be15335c4a0c7f57caf"
BASE_REVISION = "2fd448dce9a30b38cf4f106086bd7e2f13781f03"
BASE_SHA256 = "14e47a090e4b99a5152bc48717ca408af3dac28141ed705260adc61e715bec29"


def prepare(checkout: Path, base_archive: Path, output: Path) -> None:
    bundle.check_revision(checkout, REVISION)
    if bundle.sha256(base_archive) != BASE_SHA256:
        raise ValueError("Expected the original canonical Shadow asset archive")
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "assets"
    with zipfile.ZipFile(base_archive) as archive:
        archive.extractall(assets)  # The pinned SHA-256 verifies this first-party archive.
    source = assets / "source"
    shutil.copytree(checkout / "shadow_hand", source / "shadow_hand")
    for side in ("left", "right"):
        original = ET.parse(source / "shadow_hand" / f"{side}_hand.xml").getroot()
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        runtime = ET.parse(assets / f"{side}.xml", parser=parser).getroot()
        runtime.insert(0, ET.Comment("Changed by prepare_shadow_assets.py: restored Menagerie forearm collision."))
        defaults = mjcf._defaults(original)
        mesh = next(m for m in original.findall("asset/mesh") if Path(m.attrib["file"]).stem == "forearm_collision")
        attrs = mjcf._attrs(mesh, "main", defaults)
        source_mesh_dir = bundle.required(original, "compiler").get("meshdir", "")
        source_file = source / "shadow_hand" / source_mesh_dir / attrs["file"]
        collision = trimesh.load_scene(source_file, process=False).to_geometry()
        if not isinstance(collision, trimesh.Trimesh):
            raise TypeError(f"Expected triangular mesh: {source_file}")
        mesh_dir = assets / bundle.required(runtime, "compiler").get("meshdir", "")
        collision.export(mesh_dir / "forearm_collision.STL")
        ET.SubElement(
            bundle.required(runtime, "asset"),
            "mesh",
            name="forearm_collision",
            file="forearm_collision.STL",
            scale=attrs.get("scale", "1 1 1"),
        )
        source_body = bundle.required(original, "worldbody/body")
        body = bundle.required(runtime, "worldbody/body")
        geom = copy.deepcopy(next(g for g in source_body.findall("geom") if g.get("mesh") == "forearm_collision"))
        following = next(g for g in body.findall("geom") if g.get("class") == geom.get("class"))
        body.insert(list(body).index(following), geom)
        bundle.write_xml(runtime, assets / f"{side}.xml")
    notice = (
        f"# Shadow Hand E3M5 assets\n\nSource: {UPSTREAM}/tree/{REVISION}/shadow_hand.\n\n"
        "Original assets: Copyright 2022 Shadow Robot Company Ltd.; Apache-2.0. Original descriptions, meshes and license "
        "remain unchanged under source/. The original canonical robot-models bundle converts OBJ to STL, "
        "bakes mesh scales, renames XML/mesh paths and omits scene/keyframe assets. This update retains all "
        "existing visual mesh bytes, joint definitions and contact parameters, and restores the source "
        "forearm_collision geom on both hands. Its OBJ mesh is converted to STL with the original scale "
        "recorded explicitly. All canonical Menagerie collision geoms are now present; native primitives "
        "remain analytic in MuJoCo.\n\n" + (source / "shadow_hand/LICENSE").read_text()
    )
    bundle.write_bundle(
        assets,
        {
            "robot": "shadow",
            "asset_version": 2,
            "upstream": UPSTREAM,
            "revision": REVISION,
            "base_asset_revision": BASE_REVISION,
            "base_asset_sha256": BASE_SHA256,
            "descriptions": {side: f"source/shadow_hand/{side}_hand.xml" for side in ("left", "right")},
            "licenses": [{"spdx_id": "Apache-2.0", "text": "source/shadow_hand/LICENSE"}],
            "collision_policy": "complete canonical Menagerie collision definitions, including the forearm mesh",
            "runtime_scope": "canonical kinematics and visuals preserved; original Menagerie collisions restored",
            "source_to_model_basis": bundle.RUNTIME_METADATA["source_to_model_basis"],
        },
        notice,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("base_archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.checkout, args.base_archive, args.output)
