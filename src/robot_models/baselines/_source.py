"""Access original descriptions and annotations without changing baseline algorithms."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_models import _assets as assets

BASELINE_NAMES = (
    "barrett-gendexgrasp",
    "robotiq-3finger-gendexgrasp",
    "ezgripper-gendexgrasp",
    "shadow-gendexgrasp",
    "shadow-dexgraspnet",
    "allegro-gendexgrasp",  # Source-only; the runtime model remains allegro(side="left").
)


@dataclass(frozen=True)
class SourceAssets:
    """Original files in native coordinates, with their provenance manifest.

    GenDexGrasp contact annotations contain original mesh vertex indices, not
    point coordinates. DexGraspNet annotations are native body-local points.
    ``root`` is the working directory expected by original relative paths.
    """

    root: Path
    description: Path
    mesh_dir: Path
    annotations: dict[str, Path]
    manifest: dict[str, Any]

    def verify(self) -> None:
        """Check every bundled source file against the preparation-time SHA-256."""
        for relative, expected in self.manifest["source_sha256"].items():
            path = (self.root.parent / relative).resolve()
            if not path.is_relative_to(self.root.resolve()):
                raise ValueError(f"Source hash path escapes source directory: {relative!r}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Source asset hash mismatch: {path}")


def source_assets(name: str, *, model_path: Path | str | None = None) -> SourceAssets:
    """Resolve an exact source bundle using the usual path/config/cache conventions."""
    if name not in BASELINE_NAMES:
        raise ValueError(f"Unknown baseline source variant: {name!r}")
    if name == "allegro-gendexgrasp" and model_path is None:
        raise ValueError("The Allegro source-only bundle requires an explicit model_path")
    directory = assets.model_dir(name, model_path)
    manifest = json.loads((directory / "source.json").read_text())

    def path(relative: str) -> Path:
        candidate = (directory / relative).resolve()
        if not candidate.is_relative_to(directory.resolve()):
            raise ValueError(f"Source path escapes asset directory: {relative!r}")
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    return SourceAssets(
        root=path("source"),
        description=path(manifest["description"]),
        mesh_dir=path(manifest["mesh_dir"]),
        annotations={key: path(value) for key, value in manifest["annotations"].items()},
        manifest=manifest,
    )
