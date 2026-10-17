"""Check source identity before interpreting a Shadow bundle's coordinates."""

import json
from pathlib import Path

UPSTREAMS = {
    "gendexgrasp": "https://github.com/tengyu-liu/GenDexGrasp",
    "dexgraspnet": "https://github.com/PKU-EPIC/DexGraspNet",
}


def validate_source(directory: Path, source: str) -> None:
    manifest_path = directory / "source.json"
    if source == "menagerie":
        if manifest_path.exists():
            raise ValueError("Shadow source mismatch: a native source bundle requires an explicit matching source")
        return
    if not manifest_path.is_file():
        raise ValueError(f"Shadow source {source!r} requires source.json: {directory}")
    manifest = json.loads(manifest_path.read_text())
    matches = manifest.get("upstream") == UPSTREAMS[source]
    if source == "gendexgrasp":
        matches = matches and manifest.get("robot") == "shadowhand"
    if not matches:
        raise ValueError(f"Shadow source mismatch: {directory} does not identify {source!r} Shadow assets")
