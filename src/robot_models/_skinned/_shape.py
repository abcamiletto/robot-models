"""A skinned robot's rigid description at Body Models shape coefficients."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from robot_models._common import mjcf
from robot_models._common.rigid import RigidAssets


@dataclass(frozen=True)
class Shaped:
    """A variant's rigid assets at one shape, its root joint's rest position in Body Models' frame,
    the shape coefficients and its tissue density (kg/m³)."""

    assets: RigidAssets
    root: np.ndarray
    shape: np.ndarray
    density: float


def shaped(xml_path: Path, shape: Sequence[float] | None) -> Shaped:
    """Move the joint offsets, link and piece vertices of a built variant to ``shape`` (the mean by default).

    Every rest quantity is linear in the shape coefficients; ``{variant}.npz`` holds their directions.
    """
    rest = mjcf.load(xml_path)
    with np.load(xml_path.with_suffix(".npz")) as data:
        dirs = {key: data[key] for key in ("offset_dirs", "visual_dirs", "collision_dirs", "root_dirs")}
        root, density = data["root"], float(data["density"])
    count = len(dirs["offset_dirs"])
    coefficients = np.zeros(count) if shape is None else np.asarray(shape, np.float64)
    if coefficients.shape != (count,):
        raise ValueError(f"shape must have {count} coefficients, got shape {coefficients.shape}")

    def move(values: np.ndarray, key: str) -> np.ndarray:
        return (values + np.tensordot(coefficients, dirs[key], axes=1)).astype(np.float32)

    assets = replace(
        rest,
        local_offsets=move(rest.local_offsets, "offset_dirs"),
        visual=replace(rest.visual, vertices=move(rest.visual.vertices, "visual_dirs")),
        collision=replace(rest.collision, vertices=move(rest.collision.vertices, "collision_dirs")),
    )
    return Shaped(assets, move(root, "root_dirs"), coefficients.astype(np.float32), density)
