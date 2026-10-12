"""Prepare Sharpa Wave assets for the robot-models Hugging Face repository.

Use a checkout of sharpa-robotics/sharpa-urdf-usd-xml at UPSTREAM_REVISION.
The output contains assets.zip and LICENSE.md, ready for the sharpa/ hub folder.
"""

import argparse
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

UPSTREAM_REPO = "https://github.com/sharpa-robotics/sharpa-urdf-usd-xml"
UPSTREAM_REVISION = "0d19cac602f46456b819e4b6a2c09a74982c9a3e"


def prepare(source: Path, output: Path) -> None:
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected upstream revision {UPSTREAM_REVISION}, got {revision}")
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as temporary:
        assets = Path(temporary)
        for side in ("left", "right"):
            model_dir = source / "wave_01" / f"{side}_sharpa_wave"
            xml = (model_dir / f"{side}_sharpa_wave.xml").read_text()
            (assets / f"{side}.xml").write_text(xml.replace('meshdir="meshes/"', f'meshdir="{side}/meshes/"'))
            meshes = assets / side / "meshes"
            meshes.mkdir(parents=True)
            for mesh in ET.fromstring(xml).findall("asset/mesh"):
                filename = mesh.attrib["file"]
                shutil.copyfile(model_dir / "meshes" / filename, meshes / filename)
        for name in ("LICENSE.txt", "NOTICE.txt"):
            shutil.copyfile(source / name, assets / name)
        license_text = (
            f"# Sharpa Wave assets\n\nSource: {UPSTREAM_REPO}\n\nRevision: `{UPSTREAM_REVISION}`\n\n"
            "Changes: renamed the standard hand XML files to `left.xml` and `right.xml`, "
            "and changed their mesh directories to separate left and right meshes. "
            "Joint definitions, transforms, geometry, and dynamics are unchanged.\n\n"
            "## Upstream license\n\n"
            + (source / "LICENSE.txt").read_text()
            + "\n## Upstream notices\n\n"
            + (source / "NOTICE.txt").read_text()
        )
        (assets / "LICENSE.md").write_text(license_text)
        (output / "LICENSE.md").write_text(license_text)
        with zipfile.ZipFile(output / "assets.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(assets.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(assets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Pinned upstream Git checkout")
    parser.add_argument("output", type=Path, help="New directory for the hub files")
    args = parser.parse_args()
    prepare(args.source, args.output)
