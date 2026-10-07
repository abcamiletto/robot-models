"""I/O utilities for the Shadow Dexterous Hand model."""

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
# (joint index, mesh file, geom position, geom rotation)
MeshGeom = tuple[int, str, Float[np.ndarray, "3"], Float[np.ndarray, "3 3"]]


@dataclass(frozen=True)
class ShadowAssets(RigidAssets):
    actuated_joint_indices: list[int]
    actuated_joint_axes: Float[Array, "Q 3"]


def get_model_path(model_path: PathLike | None = None) -> Path:
    """Resolve a Shadow asset directory containing MuJoCo XML and STL files."""
    if model_path is None:
        model_path = config.get_model_path("shadow")
    if model_path is not None:
        return validate_path(model_path)
    cache_path = get_cache_dir() / "shadow"
    if _has_model(cache_path):
        return cache_path
    return download_model()


def download_model(output_dir: PathLike | None = None) -> Path:
    """Download the Shadow Dexterous Hand model assets."""
    output_dir = Path(output_dir) if output_dir is not None else get_cache_dir() / "shadow"
    if _has_model(output_dir):
        return validate_path(output_dir)
    print(f"Downloading Shadow model to {output_dir}...")
    download_hf_archive("shadow/assets.zip", output_dir)
    print("Done")
    return validate_path(output_dir)


def validate_path(path: PathLike) -> Path:
    path = Path(path)
    if path.is_file():
        raise ValueError(f"Expected a Shadow asset directory, got file: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Shadow model directory not found: {path}")
    for side in VALID_SIDES:
        if not (path / f"{side}.xml").exists():
            raise FileNotFoundError(f"Shadow XML not found: {path / f'{side}.xml'}")
        if not (path / "meshes" / side).is_dir():
            raise FileNotFoundError(f"Shadow mesh directory not found: {path / 'meshes' / side}")
    return path


def load_model_data(model_path: PathLike | None = None, *, side: Side = "right", dtype=np.float32) -> ShadowAssets:
    if side not in VALID_SIDES:
        raise ValueError(f"Invalid Shadow side: {side}")
    model_dir = get_model_path(model_path)
    root = ET.parse(model_dir / f"{side}.xml").getroot()
    names, parents, offsets, rotations, joints, mesh_geoms = _parse_bodies(root, side)
    joint_indices, joint_axes, joint_limits = joints
    vertices, faces, link_data = _load_link_meshes(model_dir / "meshes" / side, mesh_geoms, names, dtype=dtype)
    return ShadowAssets(
        joint_names=names,
        parents=parents,
        local_offsets=offsets.astype(dtype),
        rest_local_rotations=rotations.astype(dtype),
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
        actuated_joint_indices=joint_indices,
        actuated_joint_axes=joint_axes.astype(dtype),
        actuated_joint_limits=joint_limits.astype(dtype),
        actuated_joint_names=[names[i] for i in joint_indices],
    )


def _parse_bodies(
    root: ET.Element,
    side: str,
) -> tuple[
    list[str],
    list[int],
    Float[np.ndarray, "J 3"],
    Float[np.ndarray, "J 3 3"],
    tuple[list[int], Float[np.ndarray, "Q 3"], Float[np.ndarray, "Q 2"]],
    list[MeshGeom],
]:
    """Walk the body tree; every body is a skeleton joint with at most one hinge."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Shadow XML is missing a worldbody")
    class_axes, class_limits = mjcf.joint_defaults(root)
    mesh_file_by_name = mjcf.mesh_files_by_name(root)
    names: list[str] = []
    parents: list[int] = []
    offsets: list[Float[np.ndarray, "3"]] = []
    rotations: list[Float[np.ndarray, "3 3"]] = []
    joint_indices: list[int] = []
    joint_axes: list[Float[np.ndarray, "3"]] = []
    joint_limits: list[tuple[float, float]] = []
    mesh_geoms: list[MeshGeom] = []

    def walk(body: ET.Element, parent: int) -> None:
        index = len(names)
        # Upstream body names carry an "rh_" or "lh_" prefix.
        names.append(f"{side}_{body.get('name', '')[3:]}_skel")
        parents.append(parent)
        offsets.append(_MUJOCO_TO_MODEL @ mjcf.parse_vec(body.get("pos"), default=np.zeros(3, np.float32), size=3))
        rotations.append(_MUJOCO_TO_MODEL @ mjcf.parse_orientation(body) @ _MUJOCO_TO_MODEL.T)
        for joint in body.findall("joint"):
            # Joint classes without an axis inherit the root hand class axis.
            class_name = joint.get("class", "")
            default_axis = class_axes.get(class_name, class_axes[f"{side}_hand"])
            axis = _MUJOCO_TO_MODEL @ mjcf.parse_vec(joint.get("axis"), default=default_axis, size=3)
            joint_indices.append(index)
            joint_axes.append(axis / np.linalg.norm(axis))
            joint_limits.append(class_limits[class_name])
        for geom in body.findall("geom"):
            # Collision geoms reuse some visual meshes; keep only the visual ones.
            if geom.get("class") != "plastic_visual":
                continue
            pos = mjcf.parse_vec(geom.get("pos"), default=np.zeros(3, np.float32), size=3)
            rot = mjcf.parse_orientation(geom)
            mesh_file = mesh_file_by_name[geom.get("mesh", "")]
            mesh_geoms.append((index, mesh_file, _MUJOCO_TO_MODEL @ pos, _MUJOCO_TO_MODEL @ rot @ _MUJOCO_TO_MODEL.T))
        for child in body.findall("body"):
            walk(child, index)

    for body in worldbody.findall("body"):
        walk(body, -1)
    joints = (joint_indices, np.asarray(joint_axes), np.asarray(joint_limits))
    return names, parents, np.asarray(offsets), np.asarray(rotations), joints, mesh_geoms


def _load_link_meshes(
    mesh_dir: Path,
    mesh_geoms: list[MeshGeom],
    joint_names: list[str],
    *,
    dtype,
) -> tuple[Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"], dict[str, Any]]:
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
    for joint_index, mesh_file, geom_pos, geom_rot in mesh_geoms:
        path = mesh_dir / mesh_file
        if not path.exists():
            raise FileNotFoundError(f"Shadow mesh not found: {path}")
        vertices, faces = stl.load_stl_mesh(path, coord=_MUJOCO_TO_MODEL, dtype=dtype)
        vertices_by_link.append(vertices)
        faces_by_link.append(faces + vertex_offset)
        link_data["joint_indices"].append(joint_index)
        link_data["vertex_starts"].append(vertex_offset)
        link_data["vertex_counts"].append(vertices.shape[0])
        link_data["face_starts"].append(face_offset)
        link_data["face_counts"].append(faces.shape[0])
        link_data["geom_positions"].append(geom_pos)
        link_data["geom_rotations"].append(geom_rot)
        link_data["names"].append(f"{joint_names[joint_index].removesuffix('_skel')}/{mesh_file}")
        vertex_offset += vertices.shape[0]
        face_offset += faces.shape[0]
    if not vertices_by_link:
        raise FileNotFoundError(f"No Shadow STL link meshes found in {mesh_dir}")
    link_data["geom_positions"] = np.asarray(link_data["geom_positions"])
    link_data["geom_rotations"] = np.asarray(link_data["geom_rotations"])
    return np.concatenate(vertices_by_link), np.concatenate(faces_by_link), link_data


def _has_model(path: Path) -> bool:
    return all((path / f"{side}.xml").exists() and (path / "meshes" / side).is_dir() for side in VALID_SIDES)
