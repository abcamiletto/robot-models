"""I/O utilities for the PSYONIC Ability Hand model."""

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
JOINT_SUFFIXES = [
    "base_skel",
    "thumb_cmc_skel",
    "thumb_mcp_skel",
    "index_mcp_skel",
    "index_pip_skel",
    "middle_mcp_skel",
    "middle_pip_skel",
    "ring_mcp_skel",
    "ring_pip_skel",
    "pinky_mcp_skel",
    "pinky_pip_skel",
]
PARENTS = [-1, 0, 1, 0, 3, 0, 5, 0, 7, 0, 9]


@dataclass(frozen=True)
class AbilityAssets(RigidAssets):
    side: Side
    actuated_joint_indices: list[int]
    actuated_joint_axes: Float[Array, "Q 3"]
    coupled_joint_indices: list[int]
    coupled_joint_axes: Float[Array, "C 3"]
    coupled_driver_indices: list[int]
    coupled_polycoef: Float[Array, "C 5"]


@dataclass(frozen=True)
class _Hinge:
    index: int
    axis: Float[np.ndarray, "3"]
    limit: tuple[float, float]


def get_model_path(model_path: PathLike | None = None) -> Path:
    """Resolve an Ability Hand asset directory containing MuJoCo XML and STL files."""
    if model_path is None:
        model_path = config.get_model_path("ability")
    if model_path is not None:
        return validate_path(model_path)
    cache_path = get_cache_dir() / "ability"
    if _has_model(cache_path):
        return cache_path
    return download_model()


def download_model(output_dir: PathLike | None = None) -> Path:
    """Download the PSYONIC Ability Hand model assets."""
    output_dir = Path(output_dir) if output_dir is not None else get_cache_dir() / "ability"
    if _has_model(output_dir):
        return validate_path(output_dir)
    print(f"Downloading Ability Hand model to {output_dir}...")
    download_hf_archive("ability/assets.zip", output_dir)
    print("Done")
    return validate_path(output_dir)


def validate_path(path: PathLike) -> Path:
    path = Path(path)
    if path.is_file():
        raise ValueError(f"Expected an Ability Hand asset directory, got file: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Ability Hand model directory not found: {path}")
    for side in VALID_SIDES:
        if not (path / f"{side}.xml").exists():
            raise FileNotFoundError(f"Ability Hand XML not found: {path / f'{side}.xml'}")
        if not (path / "meshes" / side).is_dir():
            raise FileNotFoundError(f"Ability Hand mesh directory not found: {path / 'meshes' / side}")
    return path


def load_model_data(model_path: PathLike | None = None, *, side: Side = "right", dtype=np.float32) -> AbilityAssets:
    if side not in VALID_SIDES:
        raise ValueError(f"Invalid Ability Hand side: {side}")
    model_dir = get_model_path(model_path)
    root = ET.parse(model_dir / f"{side}.xml").getroot()
    names = [f"{side}_{suffix}" for suffix in JOINT_SUFFIXES]
    local_offsets, rest_local_rotations, hinges, geoms = _parse_bodies(root, names)

    # Joint equalities are joint1 = poly(joint2); joint1 is the coupled distal joint.
    couplings = {
        _skel_name(eq.get("joint1", "")): (_skel_name(eq.get("joint2", "")), eq.get("polycoef"))
        for eq in root.findall("equality/joint")
    }
    actuated_names = [name for name in hinges if name not in couplings]
    coupled_names = list(couplings)
    vertices, faces, link_data = _load_link_meshes(model_dir / "meshes" / side, geoms, dtype=dtype)
    return AbilityAssets(
        side=side,
        joint_names=names,
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
        actuated_joint_indices=[hinges[name].index for name in actuated_names],
        actuated_joint_axes=np.asarray([hinges[name].axis for name in actuated_names], dtype=dtype),
        actuated_joint_limits=np.asarray([hinges[name].limit for name in actuated_names], dtype=dtype),
        actuated_joint_names=actuated_names,
        coupled_joint_indices=[hinges[name].index for name in coupled_names],
        coupled_joint_axes=np.asarray([hinges[name].axis for name in coupled_names], dtype=dtype),
        coupled_driver_indices=[actuated_names.index(couplings[name][0]) for name in coupled_names],
        coupled_polycoef=np.asarray(
            [
                mjcf.parse_vec(couplings[name][1], default=np.zeros(5, dtype=np.float32), size=5)
                for name in coupled_names
            ],
            dtype=dtype,
        ),
    )


def _parse_bodies(
    root: ET.Element,
    names: list[str],
) -> tuple[
    Float[np.ndarray, "J 3"],
    Float[np.ndarray, "J 3 3"],
    dict[str, _Hinge],
    list[tuple[int, str, str, Float[np.ndarray, "3"], Float[np.ndarray, "3 3"]]],
]:
    by_name = {name: i for i, name in enumerate(names)}
    offsets = np.zeros((len(names), 3), dtype=np.float32)
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], len(names), axis=0)
    hinges: dict[str, _Hinge] = {}
    geoms = []
    mesh_file_by_name = mjcf.mesh_files_by_name(root)
    base = root.find("worldbody/body")
    if base is None:
        raise ValueError("Ability Hand XML is missing a root hand body")
    rotations[0] = _MUJOCO_TO_MODEL @ mjcf.parse_orientation(base) @ _MUJOCO_TO_MODEL.T

    def walk(body: ET.Element, joint_index: int) -> None:
        joint = body.find("joint")
        if joint is not None:
            name = _skel_name(joint.get("name", ""))
            joint_index = by_name[name]
            body_pos = mjcf.parse_vec(body.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
            axis = _MUJOCO_TO_MODEL @ mjcf.parse_vec(joint.get("axis"), default=np.zeros(3, dtype=np.float32), size=3)
            lo, hi = (float(x) for x in joint.get("range", "").split())
            offsets[joint_index] = _MUJOCO_TO_MODEL @ body_pos
            rotations[joint_index] = _MUJOCO_TO_MODEL @ mjcf.parse_orientation(body) @ _MUJOCO_TO_MODEL.T
            hinges[name] = _Hinge(joint_index, axis / np.linalg.norm(axis), (lo, hi))
        for geom in body.findall("geom"):
            geom_name = geom.get("name", "")
            if not geom_name.endswith("_visual"):
                continue
            pos = mjcf.parse_vec(geom.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
            rot = mjcf.parse_orientation(geom)
            mesh_file = mesh_file_by_name[geom.get("mesh", "")]
            transform = (_MUJOCO_TO_MODEL @ pos, _MUJOCO_TO_MODEL @ rot @ _MUJOCO_TO_MODEL.T)
            geoms.append((joint_index, geom_name, mesh_file, *transform))
        for child in body.findall("body"):
            walk(child, joint_index)

    walk(base, 0)
    return offsets, rotations, hinges, geoms


def _load_link_meshes(
    mesh_dir: Path,
    geoms: list[tuple[int, str, str, Float[np.ndarray, "3"], Float[np.ndarray, "3 3"]]],
    *,
    dtype,
) -> tuple[Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"], dict[str, Any]]:
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
    for joint_index, geom_name, mesh_file, geom_pos, geom_rot in geoms:
        if mesh_file not in meshes:
            path = mesh_dir / mesh_file
            if not path.exists():
                raise FileNotFoundError(f"Ability Hand mesh not found: {path}")
            meshes[mesh_file] = stl.load_stl_mesh(path, coord=_MUJOCO_TO_MODEL, dtype=dtype)
        vertices, faces = meshes[mesh_file]
        vertices_by_link.append(vertices)
        faces_by_link.append(faces + vertex_offset)
        link_data["joint_indices"].append(joint_index)
        link_data["vertex_starts"].append(vertex_offset)
        link_data["vertex_counts"].append(vertices.shape[0])
        link_data["face_starts"].append(face_offset)
        link_data["face_counts"].append(faces.shape[0])
        link_data["geom_positions"].append(geom_pos)
        link_data["geom_rotations"].append(geom_rot)
        link_data["names"].append(geom_name)
        vertex_offset += vertices.shape[0]
        face_offset += faces.shape[0]
    if not vertices_by_link:
        raise FileNotFoundError(f"No Ability Hand STL link meshes found in {mesh_dir}")
    link_data["geom_positions"] = np.asarray(link_data["geom_positions"])
    link_data["geom_rotations"] = np.asarray(link_data["geom_rotations"])
    return np.concatenate(vertices_by_link), np.concatenate(faces_by_link), link_data


def _skel_name(joint_name: str) -> str:
    return joint_name.removesuffix("_joint") + "_skel"


def _has_model(path: Path) -> bool:
    return all((path / f"{side}.xml").exists() and (path / "meshes" / side).is_dir() for side in VALID_SIDES)
