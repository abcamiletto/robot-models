"""I/O utilities for the PAL Robotics TALOS rigid model."""

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

# Skeleton joints follow MJCF joint names with the side moved to the front, e.g.
# ``arm_left_1_joint`` becomes ``left_arm_1_skel``. ``head``, ``left_foot``, and
# ``right_foot`` are leaf joints at the MJCF sites with the same names.
JOINT_NAMES = [
    "base_skel",
    "torso_1_skel",
    "torso_2_skel",
    "head_1_skel",
    "head_2_skel",
    "head",
    "left_arm_1_skel",
    "left_arm_2_skel",
    "left_arm_3_skel",
    "left_arm_4_skel",
    "left_arm_5_skel",
    "left_arm_6_skel",
    "left_arm_7_skel",
    "left_gripper_skel",
    "left_gripper_inner_double_skel",
    "left_gripper_fingertip_1_skel",
    "left_gripper_fingertip_2_skel",
    "left_gripper_motor_single_skel",
    "left_gripper_inner_single_skel",
    "left_gripper_fingertip_3_skel",
    "right_arm_1_skel",
    "right_arm_2_skel",
    "right_arm_3_skel",
    "right_arm_4_skel",
    "right_arm_5_skel",
    "right_arm_6_skel",
    "right_arm_7_skel",
    "right_gripper_skel",
    "right_gripper_inner_double_skel",
    "right_gripper_fingertip_1_skel",
    "right_gripper_fingertip_2_skel",
    "right_gripper_motor_single_skel",
    "right_gripper_inner_single_skel",
    "right_gripper_fingertip_3_skel",
    "left_leg_1_skel",
    "left_leg_2_skel",
    "left_leg_3_skel",
    "left_leg_4_skel",
    "left_leg_5_skel",
    "left_leg_6_skel",
    "left_foot",
    "right_leg_1_skel",
    "right_leg_2_skel",
    "right_leg_3_skel",
    "right_leg_4_skel",
    "right_leg_5_skel",
    "right_leg_6_skel",
    "right_foot",
]

PARENTS = [
    -1,
    0,
    1,
    2,
    3,
    4,
    2,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    12,
    14,
    14,
    12,
    12,
    18,
    2,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    26,
    28,
    28,
    26,
    26,
    32,
    0,
    34,
    35,
    36,
    37,
    38,
    39,
    0,
    41,
    42,
    43,
    44,
    45,
    46,
]


@dataclass(frozen=True)
class TalosAssets(RigidAssets):
    actuated_joint_indices: list[int]
    actuated_joint_axes: Float[Array, "Q 3"]


def get_model_path(model_path: PathLike | None = None) -> Path:
    """Resolve the TALOS XML file."""
    if model_path is None:
        model_path = config.get_model_path("talos")
    if model_path is not None:
        return validate_path(model_path)

    cache_xml = get_cache_dir() / "talos" / "talos.xml"
    if cache_xml.is_file():
        return cache_xml
    return download_model()


def download_model(output_dir: PathLike | None = None) -> Path:
    """Download TALOS XML and STL assets from Hugging Face."""
    output_dir = Path(output_dir) if output_dir is not None else get_cache_dir() / "talos"
    model_path = output_dir / "talos.xml"
    if model_path.is_file():
        return validate_path(model_path)
    print(f"Downloading TALOS model to {output_dir}...")
    download_hf_archive("talos/assets.zip", output_dir)
    print("Done")
    return validate_path(model_path)


def validate_path(path: PathLike) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "talos.xml"
    if path.suffix.lower() != ".xml":
        raise ValueError(f"Expected a TALOS XML file, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"TALOS XML not found: {path}")
    return path


def load_model_data(
    model_path: PathLike | None = None,
    *,
    convention: Convention = "soma",
    dtype=np.float32,
) -> TalosAssets:
    if convention not in VALID_CONVENTIONS:
        raise ValueError(f"Invalid convention: {convention}")
    coord = _MUJOCO_TO_MODEL if convention == "soma" else np.eye(3, dtype=np.float32)
    xml_path = get_model_path(model_path)
    root = ET.parse(xml_path).getroot()

    local_offsets, rest_local_rotations, mesh_geoms = _parse_bodies(root, coord)
    actuated_joint_indices, actuated_joint_axes, actuated_joint_limits, actuated_joint_names = _parse_actuated_joints(
        root,
        coord,
    )
    mesh_base = mjcf.mesh_base_dir(root, xml_path)
    vertices, faces, link_data = _load_link_meshes(root, mesh_base, mesh_geoms, coord, dtype=dtype)
    return TalosAssets(
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
    root: ET.Element,
    coord: Float[np.ndarray, "3 3"],
) -> tuple[
    Float[np.ndarray, "J 3"],
    Float[np.ndarray, "J 3 3"],
    list[tuple[int, str, Float[np.ndarray, "3"], Float[np.ndarray, "3 3"], str]],
]:
    """Parse joint rest transforms and visual mesh geoms from the body tree."""
    local_offsets = np.zeros((len(JOINT_NAMES), 3), dtype=np.float32)
    rest_local_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], len(JOINT_NAMES), axis=0)
    mesh_geoms: list[tuple[int, str, Float[np.ndarray, "3"], Float[np.ndarray, "3 3"], str]] = []
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("talos.xml is missing a worldbody")

    by_name = {name: i for i, name in enumerate(JOINT_NAMES)}

    def walk(body: ET.Element) -> None:
        idx = by_name[_body_to_joint_name(body)]
        if idx != 0:
            local_offsets[idx] = coord @ mjcf.parse_vec(body.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
        rest_local_rotations[idx] = coord @ mjcf.parse_orientation(body) @ coord.T
        for site in body.findall("site"):
            site_name = site.get("name", "")
            if site_name in by_name:
                site_pos = mjcf.parse_vec(site.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
                local_offsets[by_name[site_name]] = coord @ site_pos
        for geom in body.findall("geom"):
            mesh_name = geom.get("mesh")
            if mesh_name is None or geom.get("class") != "visual":
                continue
            pos = mjcf.parse_vec(geom.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
            rot = mjcf.parse_orientation(geom)
            mesh_geoms.append((idx, mesh_name, coord @ pos, coord @ rot @ coord.T, f"{body.get('name')}/{mesh_name}"))

        for child in body.findall("body"):
            walk(child)

    for body in worldbody.findall("body"):
        walk(body)
    return local_offsets, rest_local_rotations, mesh_geoms


def _parse_actuated_joints(
    root: ET.Element,
    coord: Float[np.ndarray, "3 3"],
) -> tuple[list[int], Float[np.ndarray, "Q 3"], Float[np.ndarray, "Q 2"], list[str]]:
    indices: list[int] = []
    axes: list[Float[np.ndarray, "3"]] = []
    limits: list[tuple[float, float]] = []
    names: list[str] = []
    by_name = {name: i for i, name in enumerate(JOINT_NAMES)}
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("talos.xml is missing a worldbody")

    # Document order matches MuJoCo qpos order; the free root joint is a <freejoint> tag.
    for joint in worldbody.findall(".//joint"):
        skel_name = _skel_name(joint.get("name", ""))
        axis_k = coord @ mjcf.parse_vec(joint.get("axis"), default=np.array([0.0, 0.0, 1.0], dtype=np.float32), size=3)
        axes.append(axis_k / np.linalg.norm(axis_k))
        indices.append(by_name[skel_name])
        limits.append(_joint_limit(joint))
        names.append(skel_name)
    return indices, np.asarray(axes), np.asarray(limits), names


def _load_link_meshes(
    root: ET.Element,
    mesh_base: Path,
    mesh_geoms: list[tuple[int, str, Float[np.ndarray, "3"], Float[np.ndarray, "3 3"], str]],
    coord: Float[np.ndarray, "3 3"],
    *,
    dtype,
) -> tuple:
    mesh_assets = _mesh_assets(root)
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

    for joint_idx, mesh_name, geom_pos, geom_rot, link_name in mesh_geoms:
        if mesh_name not in mesh_assets:
            raise FileNotFoundError(f"TALOS XML references missing mesh asset: {mesh_name}")
        mesh_file, scale = mesh_assets[mesh_name]
        mesh_path = mesh_base / mesh_file
        if not mesh_path.exists():
            raise FileNotFoundError(f"TALOS mesh not found: {mesh_path}")
        vertices, faces = stl.load_stl_mesh(mesh_path, coord=coord, dtype=dtype, scale=scale)
        vertices_by_link.append(vertices)
        faces_by_link.append(faces + vertex_offset)
        joint_indices.append(joint_idx)
        vertex_starts.append(vertex_offset)
        vertex_counts.append(vertices.shape[0])
        face_starts.append(face_offset)
        face_counts.append(faces.shape[0])
        geom_positions.append(geom_pos)
        geom_rotations.append(geom_rot)
        names.append(link_name)
        vertex_offset += vertices.shape[0]
        face_offset += faces.shape[0]

    if not vertices_by_link:
        raise FileNotFoundError("No TALOS STL link meshes found")
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


def _mesh_assets(root: ET.Element) -> dict[str, tuple[str, Float[np.ndarray, "3"]]]:
    """Map mesh asset names to files and scales; unnamed meshes use the file stem."""
    out = {}
    for mesh in root.findall(".//asset/mesh"):
        file = mesh.get("file")
        if not file:
            raise ValueError("<asset><mesh> entries must define a file")
        scale = mjcf.parse_vec(mesh.get("scale"), default=np.ones(3, dtype=np.float32), size=3)
        out[mesh.get("name") or Path(file).stem] = (file, scale)
    return out


def _body_to_joint_name(body: ET.Element) -> str:
    joint = body.find("joint")
    return _skel_name(joint.get("name", "") if joint is not None else body.get("name", ""))


def _skel_name(name: str) -> str:
    stem = name.removesuffix("_joint").removesuffix("_link")
    for side in ("left", "right"):
        if f"_{side}" in stem:
            stem = f"{side}_{stem.replace(f'_{side}', '', 1)}"
    return f"{stem}_skel"


def _joint_limit(joint: ET.Element) -> tuple[float, float]:
    limit = joint.get("range")
    if limit:
        lo, hi = [float(x) for x in limit.split()]
        return lo, hi
    return -np.inf, np.inf
