"""Small MJCF parsing helpers used by robot model loaders."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from jaxtyping import Float
from nanomanifold import SO3


def parse_vec(
    value: str | None,
    *,
    default: Float[np.ndarray, "N"],
    size: int | None = None,
) -> Float[np.ndarray, "N"]:
    if value is None:
        parsed = default.copy()
    else:
        parsed = np.asarray([float(x) for x in value.split()], dtype=default.dtype)
    if size is not None and parsed.shape != (size,):
        raise ValueError(f"Expected vector with {size} values, got {value!r}")
    return parsed


def parse_orientation(element: ET.Element) -> Float[np.ndarray, "3 3"]:
    """Parse an MJCF orientation as a rotation matrix.

    MuJoCo's default ``eulerseq="xyz"`` is intrinsic XYZ, which nanomanifold
    represents with uppercase ``"XYZ"`` rather than lowercase extrinsic
    ``"xyz"``.
    """
    quat = element.get("quat")
    if quat:
        q = parse_vec(quat, default=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), size=4)
        return SO3.conversions.from_quat_to_rotmat(q, convention="wxyz", xp=np).astype(np.float32)

    euler = element.get("euler")
    if euler:
        angles = parse_vec(euler, default=np.zeros(3, dtype=np.float32), size=3)
        return SO3.conversions.from_euler_to_rotmat(angles, convention="XYZ", xp=np).astype(np.float32)

    return np.eye(3, dtype=np.float32)


def mesh_base_dir(root: ET.Element, xml_path: Path) -> Path:
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    return (xml_path.parent / meshdir).resolve() if meshdir else xml_path.parent.resolve()


def mesh_files_by_name(root: ET.Element) -> dict[str, str]:
    files = {}
    for mesh in root.findall(".//asset/mesh"):
        name = mesh.get("name")
        file = mesh.get("file")
        if not name or not file:
            raise ValueError("<asset><mesh> entries must define both name and file")
        files[name] = file
    return files


def joint_defaults(
    root: ET.Element,
) -> tuple[dict[str, Float[np.ndarray, "3"]], dict[str, tuple[float, float]]]:
    axes: dict[str, Float[np.ndarray, "3"]] = {}
    limits: dict[str, tuple[float, float]] = {}
    for default in root.findall(".//default"):
        class_name = default.get("class")
        joint = default.find("joint")
        if not class_name or joint is None:
            continue
        if joint.get("axis"):
            axes[class_name] = parse_vec(joint.get("axis"), default=np.zeros(3, dtype=np.float32), size=3)
        if joint.get("range"):
            lo, hi = (float(x) for x in joint.get("range", "").split())
            limits[class_name] = (lo, hi)
    return axes, limits
