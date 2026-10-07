"""I/O utilities for the Inspire RH56 robotic hand model."""

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
ACTIVE_JOINT_SUFFIXES = {
    "thumb_proximal_yaw_skel",
    "thumb_proximal_pitch_skel",
    "index_proximal_skel",
    "middle_proximal_skel",
    "ring_proximal_skel",
    "pinky_proximal_skel",
}


@dataclass(frozen=True)
class InspireAssets(RigidAssets):
    side: Side
    actuated_joint_indices: list[int]
    actuated_joint_axes: Float[Array, "Q 3"]
    coupled_joint_indices: list[int]
    coupled_joint_axes: Float[Array, "C 3"]
    coupled_driver_indices: list[int]
    coupled_polycoef: Float[Array, "C 5"]


def get_model_path(model_path: PathLike | None = None) -> Path:
    """Resolve an Inspire asset directory containing MuJoCo XML and STL files."""
    if model_path is None:
        model_path = config.get_model_path("inspire")
    if model_path is not None:
        return validate_path(model_path)
    cache_path = get_cache_dir() / "inspire"
    if _has_model(cache_path):
        return cache_path
    return download_model()


def download_model(output_dir: PathLike | None = None) -> Path:
    """Download the Inspire hand model assets."""
    output_dir = Path(output_dir) if output_dir is not None else get_cache_dir() / "inspire"
    if _has_model(output_dir):
        return validate_path(output_dir)
    print(f"Downloading Inspire model to {output_dir}...")
    download_hf_archive("inspire/assets.zip", output_dir)
    print("Done")
    return validate_path(output_dir)


def validate_path(path: PathLike) -> Path:
    path = Path(path)
    if path.is_file():
        raise ValueError(f"Expected an Inspire asset directory, got file: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Inspire model directory not found: {path}")
    for side in VALID_SIDES:
        if not (path / f"{side}.xml").exists():
            raise FileNotFoundError(f"Inspire XML not found: {path / f'{side}.xml'}")
        if not (path / "meshes" / side).is_dir():
            raise FileNotFoundError(f"Inspire mesh directory not found: {path / 'meshes' / side}")
    return path


def load_model_data(model_path: PathLike | None = None, *, side: Side = "right", dtype=np.float32) -> InspireAssets:
    if side not in VALID_SIDES:
        raise ValueError(f"Invalid Inspire side: {side}")
    model_dir = get_model_path(model_path)
    root = ET.parse(model_dir / f"{side}.xml").getroot()
    base = root.find("worldbody/body")
    if base is None:
        raise ValueError("Inspire XML is missing a root hand body")

    # Each MJCF body is one skeleton joint, named after its hinge joint; the root body is the hand base.
    names: list[str] = []
    parents: list[int] = []
    offsets: list[Float[np.ndarray, "3"]] = []
    rotations: list[Float[np.ndarray, "3 3"]] = []
    joints: list[ET.Element | None] = []
    geoms: list[tuple[int, str, ET.Element]] = []

    def walk(body: ET.Element, parent: int) -> None:
        index = len(names)
        joint = body.find("joint")
        names.append(f"{side}_base_skel" if joint is None else joint.get("name", "").replace("_joint", "_skel"))
        parents.append(parent)
        pos = mjcf.parse_vec(body.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
        offsets.append(_MUJOCO_TO_MODEL @ pos)
        rotations.append(_MUJOCO_TO_MODEL @ mjcf.parse_orientation(body) @ _MUJOCO_TO_MODEL.T)
        joints.append(joint)
        geoms.extend((index, body.get("name", ""), geom) for geom in body.findall("geom") if geom.get("mesh"))
        for child in body.findall("body"):
            walk(child, index)

    walk(base, -1)
    by_name = {name: i for i, name in enumerate(names)}

    actuated = [i for i, name in enumerate(names) if name.removeprefix(f"{side}_") in ACTIVE_JOINT_SUFFIXES]
    actuated_names = [names[i] for i in actuated]
    coupled, drivers, polycoefs = _parse_coupled_joints(root, by_name, actuated_names)
    vertices, faces, link_data = _load_link_meshes(model_dir, root, geoms, dtype=dtype)
    return InspireAssets(
        side=side,
        joint_names=names,
        parents=parents,
        local_offsets=np.asarray(offsets, dtype=dtype),
        rest_local_rotations=np.asarray(rotations, dtype=dtype),
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
        actuated_joint_indices=actuated,
        actuated_joint_axes=_joint_axes([joints[i] for i in actuated]).astype(dtype),
        actuated_joint_limits=np.asarray([_joint_limit(joints[i]) for i in actuated], dtype=dtype),
        actuated_joint_names=actuated_names,
        coupled_joint_indices=coupled,
        coupled_joint_axes=_joint_axes([joints[i] for i in coupled]).astype(dtype),
        coupled_driver_indices=drivers,
        coupled_polycoef=polycoefs.astype(dtype),
    )


def _parse_coupled_joints(
    root: ET.Element,
    by_name: dict[str, int],
    actuated_joint_names: list[str],
) -> tuple[list[int], list[int], Float[np.ndarray, "C 5"]]:
    """Read MuJoCo joint equalities, where ``joint1 = polycoef(joint2)``."""
    qpos_by_name = {name: i for i, name in enumerate(actuated_joint_names)}
    indices = []
    drivers = []
    polycoefs = []
    for equality in root.findall("equality/joint"):
        coupled_name = equality.get("joint1", "").replace("_joint", "_skel")
        driver_name = equality.get("joint2", "").replace("_joint", "_skel")
        if coupled_name not in by_name or driver_name not in qpos_by_name:
            raise ValueError(f"Unsupported Inspire joint equality: {coupled_name} <- {driver_name}")
        polycoef = mjcf.parse_vec(equality.get("polycoef"), default=np.array([0, 1, 0, 0, 0], dtype=np.float32))
        if polycoef.shape != (5,):
            raise ValueError(f"Inspire equality joint {coupled_name} must have five polycoef values")
        indices.append(by_name[coupled_name])
        drivers.append(qpos_by_name[driver_name])
        polycoefs.append(polycoef)
    return indices, drivers, np.asarray(polycoefs, dtype=np.float32).reshape(-1, 5)


def _load_link_meshes(
    model_dir: Path,
    root: ET.Element,
    geoms: list[tuple[int, str, ET.Element]],
    *,
    dtype,
) -> tuple[Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"], dict[str, Any]]:
    mesh_file_by_name = mjcf.mesh_files_by_name(root)
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
    for joint_index, body_name, geom in geoms:
        path = model_dir / mesh_file_by_name[geom.get("mesh", "")]
        if not path.exists():
            raise FileNotFoundError(f"Inspire mesh not found: {path}")
        vertices, faces = stl.load_stl_mesh(path, coord=_MUJOCO_TO_MODEL, dtype=dtype)
        pos = mjcf.parse_vec(geom.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
        vertices_by_link.append(vertices)
        faces_by_link.append(faces + vertex_offset)
        link_data["joint_indices"].append(joint_index)
        link_data["vertex_starts"].append(vertex_offset)
        link_data["vertex_counts"].append(vertices.shape[0])
        link_data["face_starts"].append(face_offset)
        link_data["face_counts"].append(faces.shape[0])
        link_data["geom_positions"].append(_MUJOCO_TO_MODEL @ pos)
        link_data["geom_rotations"].append(_MUJOCO_TO_MODEL @ mjcf.parse_orientation(geom) @ _MUJOCO_TO_MODEL.T)
        link_data["names"].append(body_name)
        vertex_offset += vertices.shape[0]
        face_offset += faces.shape[0]
    if not vertices_by_link:
        raise FileNotFoundError(f"No Inspire STL link meshes found in {model_dir}")
    link_data["geom_positions"] = np.asarray(link_data["geom_positions"])
    link_data["geom_rotations"] = np.asarray(link_data["geom_rotations"])
    return np.concatenate(vertices_by_link), np.concatenate(faces_by_link), link_data


def _joint_axes(joints: list[ET.Element | None]) -> Float[np.ndarray, "N 3"]:
    axes = []
    for joint in joints:
        if joint is None:
            raise ValueError("Inspire hinge body is missing its joint")
        axis = _MUJOCO_TO_MODEL @ mjcf.parse_vec(joint.get("axis"), default=np.array([0, 0, 1], dtype=np.float32))
        axes.append(axis / np.linalg.norm(axis))
    return np.asarray(axes, dtype=np.float32).reshape(-1, 3)


def _joint_limit(joint: ET.Element | None) -> tuple[float, float]:
    limit = joint.get("range") if joint is not None else None
    if limit is None:
        raise ValueError("Inspire actuated joint is missing its range")
    lo, hi = [float(x) for x in limit.split()]
    return lo, hi


def _has_model(path: Path) -> bool:
    return all((path / f"{side}.xml").exists() and (path / "meshes" / side).is_dir() for side in VALID_SIDES)
