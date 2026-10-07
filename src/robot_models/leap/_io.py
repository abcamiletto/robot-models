"""I/O utilities for the LEAP Hand v1 robotic hand model."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int

from robot_models import _config as config
from robot_models._cache import download_hf_archive, get_cache_dir
from robot_models._common import coordinates, mjcf, stl
from robot_models._common.rigid import RigidAssets

PathLike = Path | str
Side = Literal["left", "right"]
Array = Any
VALID_SIDES = ("left", "right")
_MUJOCO_TO_MODEL = np.asarray(coordinates.MUJOCO_Z_UP_TO_Y_UP, dtype=np.float32)


@dataclass(frozen=True)
class LeapAssets(RigidAssets):
    side: Side
    actuated_joint_indices: list[int]
    actuated_joint_axes: Float[Array, "Q 3"]


def get_model_path(model_path: PathLike | None = None) -> Path:
    """Resolve a LEAP asset directory containing MuJoCo XML and STL files."""
    if model_path is None:
        model_path = config.get_model_path("leap")
    if model_path is not None:
        return validate_path(model_path)
    cache_path = get_cache_dir() / "leap"
    if _has_model(cache_path):
        return cache_path
    return download_model()


def download_model(output_dir: PathLike | None = None) -> Path:
    """Download the LEAP Hand v1 model assets."""
    output_dir = Path(output_dir) if output_dir is not None else get_cache_dir() / "leap"
    if _has_model(output_dir):
        return validate_path(output_dir)
    print(f"Downloading LEAP model to {output_dir}...")
    download_hf_archive("leap/assets.zip", output_dir)
    print("Done")
    return validate_path(output_dir)


def validate_path(path: PathLike) -> Path:
    path = Path(path)
    if path.is_file():
        raise ValueError(f"Expected a LEAP asset directory, got file: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"LEAP model directory not found: {path}")
    for side in VALID_SIDES:
        if not (path / f"{side}.xml").exists():
            raise FileNotFoundError(f"LEAP XML not found: {path / f'{side}.xml'}")
        if not (path / "meshes" / side).is_dir():
            raise FileNotFoundError(f"LEAP mesh directory not found: {path / 'meshes' / side}")
    return path


def load_model_data(model_path: PathLike | None = None, *, side: Side = "right", dtype=np.float32) -> LeapAssets:
    if side not in VALID_SIDES:
        raise ValueError(f"Invalid LEAP side: {side}")
    xml_path = get_model_path(model_path) / f"{side}.xml"
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    palm = worldbody.find("body") if worldbody is not None else None
    if palm is None:
        raise ValueError("LEAP XML is missing a palm body")

    class_axes, class_limits = mjcf.joint_defaults(root)
    names: list[str] = []
    parents: list[int] = []
    offsets: list[Float[np.ndarray, "3"]] = []
    rotations: list[Float[np.ndarray, "3 3"]] = []
    joints: list[tuple[int, ET.Element]] = []
    link_geoms: list[tuple[int, ET.Element]] = []

    def walk(body: ET.Element, parent: int) -> None:
        index = len(names)
        joint = body.find("joint")
        names.append(f"{side}_{'palm' if joint is None else joint.get('name')}_skel")
        parents.append(parent)
        # The root keeps its rest rotation but not its MuJoCo scene placement.
        body_pos = mjcf.parse_vec(body.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
        offsets.append(_MUJOCO_TO_MODEL @ body_pos if parent >= 0 else np.zeros(3, dtype=np.float32))
        rotations.append(_MUJOCO_TO_MODEL @ mjcf.parse_orientation(body) @ _MUJOCO_TO_MODEL.T)
        if joint is not None:
            joints.append((index, joint))
        link_geoms.extend((index, geom) for geom in body.findall("geom"))
        for child in body.findall("body"):
            walk(child, index)

    walk(palm, -1)
    axes = np.asarray([_MUJOCO_TO_MODEL @ _joint_axis(joint, class_axes) for _, joint in joints])
    limits = np.asarray([_joint_limit(joint, class_limits) for _, joint in joints])
    vertices, faces, link_data = _load_link_meshes(root, xml_path, link_geoms, dtype=dtype)
    return LeapAssets(
        side=side,
        joint_names=names,
        parents=parents,
        local_offsets=np.asarray(offsets).astype(dtype),
        rest_local_rotations=np.asarray(rotations).astype(dtype),
        vertices=vertices.astype(dtype),
        faces=faces.astype(np.int64),
        link_joint_indices=link_data["joint_indices"],
        link_vertex_starts=link_data["vertex_starts"],
        link_vertex_counts=link_data["vertex_counts"],
        link_face_starts=link_data["face_starts"],
        link_face_counts=link_data["face_counts"],
        link_geom_positions=link_data["geom_positions"].astype(dtype),
        link_geom_rotations=link_data["geom_rotations"].astype(dtype),
        link_names=link_data["names"],
        actuated_joint_indices=[index for index, _ in joints],
        actuated_joint_axes=(axes / np.linalg.norm(axes, axis=-1, keepdims=True)).astype(dtype),
        actuated_joint_limits=limits.astype(dtype),
        actuated_joint_names=[names[index] for index, _ in joints],
    )


def _load_link_meshes(
    root: ET.Element,
    xml_path: Path,
    link_geoms: list[tuple[int, ET.Element]],
    *,
    dtype,
) -> tuple[Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"], dict[str, Any]]:
    mesh_base = mjcf.mesh_base_dir(root, xml_path)
    mesh_file_by_name = mjcf.mesh_files_by_name(root)
    # Fingertip geoms take their mesh from a geom class default.
    class_meshes = {
        default.get("class"): geom.get("mesh")
        for default in root.findall(".//default")
        if (geom := default.find("geom")) is not None and geom.get("mesh")
    }
    meshes: dict[str, tuple[Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"]]] = {}
    vertices_by_link = []
    faces_by_link = []
    link_data = {
        "joint_indices": [],
        "vertex_starts": [],
        "vertex_counts": [],
        "face_starts": [],
        "face_counts": [],
        "geom_positions": [],
        "geom_rotations": [],
        "names": [],
    }
    vertex_offset = 0
    face_offset = 0
    for joint_index, geom in link_geoms:
        mesh_name = geom.get("mesh") or class_meshes.get(geom.get("class"))
        if mesh_name is None:
            continue
        if mesh_name not in mesh_file_by_name:
            raise FileNotFoundError(f"LEAP XML references missing mesh asset: {mesh_name}")
        if mesh_name not in meshes:
            path = mesh_base / mesh_file_by_name[mesh_name]
            if not path.exists():
                raise FileNotFoundError(f"LEAP mesh not found: {path}")
            meshes[mesh_name] = stl.load_stl_mesh(path, coord=_MUJOCO_TO_MODEL, dtype=dtype)
        vertices, faces = meshes[mesh_name]
        pos = mjcf.parse_vec(geom.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
        rot = mjcf.parse_orientation(geom)
        vertices_by_link.append(vertices)
        faces_by_link.append(faces + vertex_offset)
        link_data["joint_indices"].append(joint_index)
        link_data["vertex_starts"].append(vertex_offset)
        link_data["vertex_counts"].append(vertices.shape[0])
        link_data["face_starts"].append(face_offset)
        link_data["face_counts"].append(faces.shape[0])
        link_data["geom_positions"].append(_MUJOCO_TO_MODEL @ pos)
        link_data["geom_rotations"].append(_MUJOCO_TO_MODEL @ rot @ _MUJOCO_TO_MODEL.T)
        link_data["names"].append(geom.get("name", mesh_name))
        vertex_offset += vertices.shape[0]
        face_offset += faces.shape[0]
    if not vertices_by_link:
        raise FileNotFoundError(f"No LEAP STL link meshes found in {mesh_base}")
    link_data["geom_positions"] = np.asarray(link_data["geom_positions"])
    link_data["geom_rotations"] = np.asarray(link_data["geom_rotations"])
    return np.concatenate(vertices_by_link), np.concatenate(faces_by_link), link_data


def _joint_axis(
    joint: ET.Element,
    class_axes: dict[str, Float[np.ndarray, "3"]],
) -> Float[np.ndarray, "3"]:
    if joint.get("axis"):
        return mjcf.parse_vec(joint.get("axis"), default=np.zeros(3, dtype=np.float32), size=3)
    class_name = joint.get("class")
    if class_name in class_axes:
        return class_axes[class_name]
    raise ValueError(f"Missing axis for LEAP joint {joint.get('name')}")


def _joint_limit(joint: ET.Element, class_limits: dict[str, tuple[float, float]]) -> tuple[float, float]:
    limit = joint.get("range")
    if limit is not None:
        lo, hi = [float(x) for x in limit.split()]
        return lo, hi
    class_name = joint.get("class")
    if class_name in class_limits:
        return class_limits[class_name]
    raise ValueError(f"Missing limit for LEAP joint {joint.get('name')}")


def _has_model(path: Path) -> bool:
    return all((path / f"{side}.xml").exists() and (path / "meshes" / side).is_dir() for side in VALID_SIDES)
