"""I/O utilities for the Unitree H1 rigid model."""

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
    "pelvis_skel",
    "left_hip_yaw_skel",
    "left_hip_roll_skel",
    "left_hip_pitch_skel",
    "left_knee_skel",
    "left_ankle_skel",
    "right_hip_yaw_skel",
    "right_hip_roll_skel",
    "right_hip_pitch_skel",
    "right_knee_skel",
    "right_ankle_skel",
    "torso_skel",
    "left_shoulder_pitch_skel",
    "left_shoulder_roll_skel",
    "left_shoulder_yaw_skel",
    "left_elbow_skel",
    "right_shoulder_pitch_skel",
    "right_shoulder_roll_skel",
    "right_shoulder_yaw_skel",
    "right_elbow_skel",
]

PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 11, 16, 17, 18]

H1_MESH_JOINT_MAP = {
    "pelvis_skel": ["pelvis.stl"],
    "left_hip_yaw_skel": ["left_hip_yaw_link.stl"],
    "left_hip_roll_skel": ["left_hip_roll_link.stl"],
    "left_hip_pitch_skel": ["left_hip_pitch_link.stl"],
    "left_knee_skel": ["left_knee_link.stl"],
    "left_ankle_skel": ["left_ankle_link.stl"],
    "right_hip_yaw_skel": ["right_hip_yaw_link.stl"],
    "right_hip_roll_skel": ["right_hip_roll_link.stl"],
    "right_hip_pitch_skel": ["right_hip_pitch_link.stl"],
    "right_knee_skel": ["right_knee_link.stl"],
    "right_ankle_skel": ["right_ankle_link.stl"],
    "torso_skel": ["torso_link.stl", "logo_link.stl"],
    "left_shoulder_pitch_skel": ["left_shoulder_pitch_link.stl"],
    "left_shoulder_roll_skel": ["left_shoulder_roll_link.stl"],
    "left_shoulder_yaw_skel": ["left_shoulder_yaw_link.stl"],
    "left_elbow_skel": ["left_elbow_link.stl"],
    "right_shoulder_pitch_skel": ["right_shoulder_pitch_link.stl"],
    "right_shoulder_roll_skel": ["right_shoulder_roll_link.stl"],
    "right_shoulder_yaw_skel": ["right_shoulder_yaw_link.stl"],
    "right_elbow_skel": ["right_elbow_link.stl"],
}


@dataclass(frozen=True)
class H1Assets(RigidAssets):
    actuated_joint_indices: list[int]
    actuated_joint_axes: Float[Array, "Q 3"]


def get_model_path(model_path: PathLike | None = None) -> Path:
    """Resolve the H1 XML file."""
    if model_path is None:
        model_path = config.get_model_path("h1")
    if model_path is not None:
        return validate_path(model_path)

    cache_xml = get_cache_dir() / "h1" / "h1.xml"
    if cache_xml.is_file():
        return cache_xml
    return download_model()


def download_model(output_dir: PathLike | None = None) -> Path:
    """Download H1 XML and STL assets from Hugging Face."""
    output_dir = Path(output_dir) if output_dir is not None else get_cache_dir() / "h1"
    model_path = output_dir / "h1.xml"
    if model_path.is_file():
        return validate_path(model_path)
    print(f"Downloading H1 model to {output_dir}...")
    download_hf_archive("h1/assets.zip", output_dir)
    print("Done")
    return validate_path(model_path)


def validate_path(path: PathLike) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "h1.xml"
    if path.suffix.lower() != ".xml":
        raise ValueError(f"Expected an H1 XML file, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"H1 XML not found: {path}")
    return path


def load_model_data(
    model_path: PathLike | None = None,
    *,
    convention: Convention = "soma",
    dtype=np.float32,
) -> H1Assets:
    if convention not in VALID_CONVENTIONS:
        raise ValueError(f"Invalid convention: {convention}")
    coord = _MUJOCO_TO_MODEL if convention == "soma" else np.eye(3, dtype=np.float32)
    xml_path = get_model_path(model_path)
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("h1.xml is missing a worldbody")

    local_offsets, rest_local_rotations = _parse_joint_rest(worldbody, coord)
    mesh_base = mjcf.mesh_base_dir(root, xml_path)
    mesh_transforms = _parse_mesh_local_transforms(root, mesh_base, coord)
    actuated_joint_indices, actuated_joint_axes, actuated_joint_limits, actuated_joint_names = _parse_actuated_joints(
        worldbody,
        coord,
    )
    vertices, faces, link_data = _load_link_meshes(mesh_transforms, coord, dtype=dtype)
    return H1Assets(
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


def _parse_joint_rest(
    worldbody: ET.Element,
    coord: Float[np.ndarray, "3 3"],
) -> tuple[Float[np.ndarray, "J 3"], Float[np.ndarray, "J 3 3"]]:
    local_offsets = np.zeros((len(JOINT_NAMES), 3), dtype=np.float32)
    rest_local_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], len(JOINT_NAMES), axis=0)
    by_name = {name: i for i, name in enumerate(JOINT_NAMES)}

    for body in worldbody.iter("body"):
        idx = by_name[_body_to_joint_name(body)]
        # The root offset comes from global_translation, as for the MuJoCo free joint.
        if idx != 0:
            body_pos = mjcf.parse_vec(body.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
            local_offsets[idx] = coord @ body_pos
        rest_local_rotations[idx] = coord @ mjcf.parse_orientation(body) @ coord.T
    return local_offsets, rest_local_rotations


def _parse_mesh_local_transforms(
    root: ET.Element,
    mesh_base: Path,
    coord: Float[np.ndarray, "3 3"],
) -> dict[str, tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3 3"], Path]]:
    mesh_file_by_name = mjcf.mesh_files_by_name(root)
    out: dict[str, tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3 3"], Path]] = {}
    for geom in root.findall(".//geom"):
        mesh_name = geom.get("mesh")
        if mesh_name is None:
            continue
        mesh_file = mesh_file_by_name.get(mesh_name)
        if mesh_file is None:
            raise FileNotFoundError(f"H1 XML references missing mesh asset: {mesh_name}")
        key = Path(mesh_file).name
        if key in out:
            continue
        pos = mjcf.parse_vec(geom.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
        rot = mjcf.parse_orientation(geom)
        mesh_path = Path(mesh_file)
        if not mesh_path.is_absolute():
            mesh_path = mesh_base / mesh_path
        out[key] = (coord @ pos, coord @ rot @ coord.T, mesh_path.resolve())
    return out


def _parse_actuated_joints(
    worldbody: ET.Element,
    coord: Float[np.ndarray, "3 3"],
) -> tuple[list[int], Float[np.ndarray, "Q 3"], Float[np.ndarray, "Q 2"], list[str]]:
    indices: list[int] = []
    axes: list[Float[np.ndarray, "3"]] = []
    limits: list[tuple[float, float]] = []
    names: list[str] = []
    by_name = {name: i for i, name in enumerate(JOINT_NAMES)}

    for joint in worldbody.iter("joint"):
        skel_name = f"{joint.get('name')}_skel"
        axis = coord @ mjcf.parse_vec(joint.get("axis"), default=np.array([0.0, 0.0, 1.0], dtype=np.float32), size=3)
        lo, hi = (float(x) for x in joint.get("range", "-inf inf").split())
        axes.append(axis / np.linalg.norm(axis))
        indices.append(by_name[skel_name])
        limits.append((lo, hi))
        names.append(skel_name)
    return indices, np.asarray(axes), np.asarray(limits), names


def _load_link_meshes(
    mesh_transforms: dict[str, tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3 3"], Path]],
    coord: Float[np.ndarray, "3 3"],
    *,
    dtype,
) -> tuple:
    vertices_by_link: list[Float[np.ndarray, "V 3"]] = []
    faces_by_link: list[Int[np.ndarray, "F 3"]] = []
    joint_indices: list[int] = []
    vertex_starts: list[int] = []
    vertex_counts: list[int] = []
    face_starts: list[int] = []
    face_counts: list[int] = []
    geom_positions: list[Float[np.ndarray, "3"]] = []
    geom_rotations: list[Float[np.ndarray, "3 3"]] = []
    names: list[str] = []
    vertex_offset = 0
    face_offset = 0
    by_name = {name: i for i, name in enumerate(JOINT_NAMES)}

    for joint_name, mesh_files in H1_MESH_JOINT_MAP.items():
        joint_idx = by_name[joint_name]
        for mesh_file in mesh_files:
            if mesh_file not in mesh_transforms:
                raise FileNotFoundError(f"H1 XML does not reference expected mesh: {mesh_file}")
            geom_pos, geom_rot, mesh_path = mesh_transforms[mesh_file]
            if not mesh_path.exists():
                raise FileNotFoundError(f"H1 mesh not found: {mesh_path}")
            vertices, faces = stl.load_stl_mesh(mesh_path, coord=coord, dtype=dtype)
            vertices_by_link.append(vertices)
            faces_by_link.append(faces + vertex_offset)
            joint_indices.append(joint_idx)
            vertex_starts.append(vertex_offset)
            vertex_counts.append(vertices.shape[0])
            face_starts.append(face_offset)
            face_counts.append(faces.shape[0])
            geom_positions.append(geom_pos)
            geom_rotations.append(geom_rot)
            names.append(mesh_file)
            vertex_offset += vertices.shape[0]
            face_offset += faces.shape[0]

    link_data = {
        "joint_indices": joint_indices,
        "vertex_starts": vertex_starts,
        "vertex_counts": vertex_counts,
        "face_starts": face_starts,
        "face_counts": face_counts,
        "geom_positions": np.asarray(geom_positions),
        "geom_rotations": np.asarray(geom_rotations),
        "names": names,
    }
    return np.concatenate(vertices_by_link), np.concatenate(faces_by_link), link_data


def _body_to_joint_name(body: ET.Element) -> str:
    joint = body.find("joint")
    if joint is not None:
        return f"{joint.get('name')}_skel"
    return f"{body.get('name')}_skel"
