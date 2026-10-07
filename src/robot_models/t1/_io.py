"""I/O utilities for the Booster T1 rigid model."""

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
Convention = Literal["soma", "mujoco"]
Array = Any

_MUJOCO_TO_MODEL = np.asarray(coordinates.MUJOCO_Z_UP_TO_Y_UP, dtype=np.float32)
VALID_CONVENTIONS = ("soma", "mujoco")

JOINT_NAMES = [
    "trunk_skel",
    "head_yaw_skel",
    "head_pitch_skel",
    "left_shoulder_pitch_skel",
    "left_shoulder_roll_skel",
    "left_elbow_pitch_skel",
    "left_elbow_yaw_skel",
    "right_shoulder_pitch_skel",
    "right_shoulder_roll_skel",
    "right_elbow_pitch_skel",
    "right_elbow_yaw_skel",
    "waist_skel",
    "left_hip_pitch_skel",
    "left_hip_roll_skel",
    "left_hip_yaw_skel",
    "left_knee_pitch_skel",
    "left_ankle_pitch_skel",
    "left_ankle_roll_skel",
    "right_hip_pitch_skel",
    "right_hip_roll_skel",
    "right_hip_yaw_skel",
    "right_knee_pitch_skel",
    "right_ankle_pitch_skel",
    "right_ankle_roll_skel",
]

PARENTS = [-1, 0, 1, 0, 3, 4, 5, 0, 7, 8, 9, 0, 11, 12, 13, 14, 15, 16, 11, 18, 19, 20, 21, 22]

MeshTransform = tuple[int, Float[np.ndarray, "3"], Float[np.ndarray, "3 3"], Path]


@dataclass(frozen=True)
class T1Assets(RigidAssets):
    actuated_joint_indices: list[int]
    actuated_joint_axes: Float[Array, "Q 3"]


def get_model_path(model_path: PathLike | None = None) -> Path:
    """Resolve the T1 XML file."""
    if model_path is None:
        model_path = config.get_model_path("t1")
    if model_path is not None:
        return validate_path(model_path)

    cache_xml = get_cache_dir() / "t1" / "t1.xml"
    if cache_xml.is_file():
        return cache_xml
    return download_model()


def download_model(output_dir: PathLike | None = None) -> Path:
    """Download T1 XML and STL assets from Hugging Face."""
    output_dir = Path(output_dir) if output_dir is not None else get_cache_dir() / "t1"
    model_path = output_dir / "t1.xml"
    if model_path.is_file():
        return validate_path(model_path)
    print(f"Downloading T1 model to {output_dir}...")
    download_hf_archive("t1/assets.zip", output_dir)
    print("Done")
    return validate_path(model_path)


def validate_path(path: PathLike) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "t1.xml"
    if path.suffix.lower() != ".xml":
        raise ValueError(f"Expected a T1 XML file, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"T1 XML not found: {path}")
    return path


def load_model_data(
    model_path: PathLike | None = None,
    *,
    convention: Convention = "soma",
    dtype=np.float32,
) -> T1Assets:
    if convention not in VALID_CONVENTIONS:
        raise ValueError(f"Invalid convention: {convention}")
    coord = _MUJOCO_TO_MODEL if convention == "soma" else np.eye(3, dtype=np.float32)
    xml_path = get_model_path(model_path)
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("t1.xml is missing a worldbody")

    local_offsets, rest_local_rotations, mesh_transforms = _parse_bodies(
        worldbody, mjcf.mesh_files_by_name(root), mjcf.mesh_base_dir(root, xml_path), coord
    )
    actuated_joint_indices, actuated_joint_axes, actuated_joint_limits, actuated_joint_names = _parse_actuated_joints(
        worldbody, coord
    )
    vertices, faces, link_data = _load_link_meshes(mesh_transforms, coord, dtype=dtype)
    return T1Assets(
        joint_names=JOINT_NAMES.copy(),
        parents=PARENTS.copy(),
        local_offsets=local_offsets.astype(dtype),
        rest_local_rotations=rest_local_rotations.astype(dtype),
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
        actuated_joint_indices=actuated_joint_indices,
        actuated_joint_axes=actuated_joint_axes.astype(dtype),
        actuated_joint_limits=actuated_joint_limits.astype(dtype),
        actuated_joint_names=actuated_joint_names,
    )


def _parse_bodies(
    worldbody: ET.Element,
    mesh_file_by_name: dict[str, str],
    mesh_base: Path,
    coord: Float[np.ndarray, "3 3"],
) -> tuple[Float[np.ndarray, "J 3"], Float[np.ndarray, "J 3 3"], list[MeshTransform]]:
    local_offsets = np.zeros((len(JOINT_NAMES), 3), dtype=np.float32)
    rest_local_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], len(JOINT_NAMES), axis=0)
    mesh_transforms: list[MeshTransform] = []
    by_name = {name: i for i, name in enumerate(JOINT_NAMES)}

    def walk(body: ET.Element) -> None:
        idx = by_name[_body_to_joint_name(body)]
        if idx != 0:
            body_pos = mjcf.parse_vec(body.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
            local_offsets[idx] = coord @ body_pos
        rest_local_rotations[idx] = coord @ mjcf.parse_orientation(body) @ coord.T

        for geom in body.findall("geom"):
            mesh_name = geom.get("mesh")
            if mesh_name is None:
                continue
            mesh_file = mesh_file_by_name.get(mesh_name)
            if mesh_file is None:
                raise FileNotFoundError(f"T1 XML references missing mesh asset: {mesh_name}")
            pos = mjcf.parse_vec(geom.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
            rot = mjcf.parse_orientation(geom)
            mesh_transforms.append((idx, coord @ pos, coord @ rot @ coord.T, mesh_base / mesh_file))

        for child in body.findall("body"):
            walk(child)

    for body in worldbody.findall("body"):
        walk(body)
    return local_offsets, rest_local_rotations, mesh_transforms


def _parse_actuated_joints(
    worldbody: ET.Element,
    coord: Float[np.ndarray, "3 3"],
) -> tuple[list[int], Float[np.ndarray, "Q 3"], Float[np.ndarray, "Q 2"], list[str]]:
    indices: list[int] = []
    axes: list[Float[np.ndarray, "3"]] = []
    limits: list[tuple[float, float]] = []
    names: list[str] = []
    by_name = {name: i for i, name in enumerate(JOINT_NAMES)}

    # Document order matches MuJoCo qpos order after the free joint.
    for joint in worldbody.findall(".//joint"):
        skel_name = _joint_to_skel_name(joint.get("name", ""))
        axis = coord @ mjcf.parse_vec(joint.get("axis"), default=np.array([0.0, 0.0, 1.0], dtype=np.float32), size=3)
        lo, hi = (float(x) for x in joint.get("range", "-inf inf").split())
        indices.append(by_name[skel_name])
        axes.append(axis / np.linalg.norm(axis))
        limits.append((lo, hi))
        names.append(skel_name)
    return indices, np.asarray(axes), np.asarray(limits), names


def _load_link_meshes(
    mesh_transforms: list[MeshTransform],
    coord: Float[np.ndarray, "3 3"],
    *,
    dtype,
) -> tuple[Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"], dict[str, Any]]:
    vertices_by_link: list[Float[np.ndarray, "V 3"]] = []
    faces_by_link: list[Int[np.ndarray, "F 3"]] = []
    link_data: dict[str, Any] = {
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

    for joint_idx, geom_pos, geom_rot, mesh_path in mesh_transforms:
        if not mesh_path.exists():
            raise FileNotFoundError(f"T1 mesh not found: {mesh_path}")
        vertices, faces = stl.load_stl_mesh(mesh_path, coord=coord, dtype=dtype)
        vertices_by_link.append(vertices)
        faces_by_link.append(faces + vertex_offset)
        link_data["joint_indices"].append(joint_idx)
        link_data["vertex_starts"].append(vertex_offset)
        link_data["vertex_counts"].append(vertices.shape[0])
        link_data["face_starts"].append(face_offset)
        link_data["face_counts"].append(faces.shape[0])
        link_data["geom_positions"].append(geom_pos)
        link_data["geom_rotations"].append(geom_rot)
        link_data["names"].append(mesh_path.name)
        vertex_offset += vertices.shape[0]
        face_offset += faces.shape[0]

    if not vertices_by_link:
        raise FileNotFoundError("No T1 STL link meshes found")
    link_data["geom_positions"] = np.asarray(link_data["geom_positions"])
    link_data["geom_rotations"] = np.asarray(link_data["geom_rotations"])
    return np.concatenate(vertices_by_link), np.concatenate(faces_by_link), link_data


def _body_to_joint_name(body: ET.Element) -> str:
    joint = body.find("joint")
    if joint is not None:
        return _joint_to_skel_name(joint.get("name", ""))
    return body.get("name", "").lower() + "_skel"


def _joint_to_skel_name(joint_name: str) -> str:
    # Upstream prefixes the head yaw joint with "AA" so that it sorts first.
    return joint_name.lower().removeprefix("aa") + "_skel"
