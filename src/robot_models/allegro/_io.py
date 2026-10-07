"""I/O utilities for the Wonik Robotics Allegro hand model."""

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
BODY_NAMES = [
    "palm",
    "ff_base",
    "ff_proximal",
    "ff_medial",
    "ff_distal",
    "mf_base",
    "mf_proximal",
    "mf_medial",
    "mf_distal",
    "rf_base",
    "rf_proximal",
    "rf_medial",
    "rf_distal",
    "th_base",
    "th_proximal",
    "th_medial",
    "th_distal",
]
PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15]


@dataclass(frozen=True)
class AllegroAssets(RigidAssets):
    actuated_joint_indices: list[int]
    actuated_joint_axes: Float[Array, "Q 3"]


@dataclass
class _ParsedBodies:
    local_offsets: Float[np.ndarray, "J 3"]
    rest_local_rotations: Float[np.ndarray, "J 3 3"]
    joint_indices: list[int]
    joint_axes: list[Float[np.ndarray, "3"]]
    joint_limits: list[Float[np.ndarray, "2"]]
    # One entry per mesh geom: joint index, mesh file, joint-local position and rotation, body name.
    links: list[tuple[int, str, Float[np.ndarray, "3"], Float[np.ndarray, "3 3"], str]]


def get_model_path(model_path: PathLike | None = None) -> Path:
    """Resolve an Allegro asset directory containing MuJoCo Menagerie XML and STL files."""
    if model_path is None:
        model_path = config.get_model_path("allegro")
    if model_path is not None:
        return validate_path(model_path)
    cache_path = get_cache_dir() / "allegro"
    if _has_model(cache_path):
        return cache_path
    return download_model()


def download_model(output_dir: PathLike | None = None) -> Path:
    """Download the Allegro hand model assets."""
    output_dir = Path(output_dir) if output_dir is not None else get_cache_dir() / "allegro"
    if _has_model(output_dir):
        return validate_path(output_dir)
    print(f"Downloading Allegro model to {output_dir}...")
    download_hf_archive("allegro/assets.zip", output_dir)
    print("Done")
    return validate_path(output_dir)


def validate_path(path: PathLike) -> Path:
    path = Path(path)
    if path.is_file():
        raise ValueError(f"Expected an Allegro asset directory, got file: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Allegro model directory not found: {path}")
    for side in VALID_SIDES:
        if not (path / f"{side}.xml").exists():
            raise FileNotFoundError(f"Allegro XML not found: {path / f'{side}.xml'}")
        if not (path / "meshes" / side).is_dir():
            raise FileNotFoundError(f"Allegro mesh directory not found: {path / 'meshes' / side}")
    return path


def load_model_data(model_path: PathLike | None = None, *, side: Side = "right", dtype=np.float32) -> AllegroAssets:
    if side not in VALID_SIDES:
        raise ValueError(f"Invalid Allegro side: {side}")
    model_dir = get_model_path(model_path)
    root = ET.parse(model_dir / f"{side}.xml").getroot()
    names = [f"{side}_{name}_skel" for name in BODY_NAMES]
    parsed = _parse_bodies(root)
    vertices, faces, link_data = _load_link_meshes(model_dir / "meshes" / side, parsed.links, side, dtype=dtype)
    return AllegroAssets(
        joint_names=names,
        parents=PARENTS.copy(),
        local_offsets=parsed.local_offsets.astype(dtype),
        rest_local_rotations=parsed.rest_local_rotations.astype(dtype),
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
        actuated_joint_indices=parsed.joint_indices,
        actuated_joint_axes=np.asarray(parsed.joint_axes).astype(dtype),
        actuated_joint_limits=np.asarray(parsed.joint_limits).astype(dtype),
        actuated_joint_names=[names[i] for i in parsed.joint_indices],
    )


def _parse_bodies(root: ET.Element) -> _ParsedBodies:
    worldbody = root.find("worldbody")
    palm = worldbody.find("body") if worldbody is not None else None
    if palm is None:
        raise ValueError("Allegro XML is missing the palm body")
    coord = _MUJOCO_TO_MODEL
    joint_defaults = _class_defaults(root, "joint")
    geom_defaults = _class_defaults(root, "geom")
    mesh_file_by_name = mjcf.mesh_files_by_name(root)
    by_name = {name: i for i, name in enumerate(BODY_NAMES)}
    parsed = _ParsedBodies(
        local_offsets=np.zeros((len(BODY_NAMES), 3), dtype=np.float32),
        rest_local_rotations=np.repeat(np.eye(3, dtype=np.float32)[None], len(BODY_NAMES), axis=0),
        joint_indices=[],
        joint_axes=[],
        joint_limits=[],
        links=[],
    )

    def walk(
        body: ET.Element,
        owner: int,
        pos: Float[np.ndarray, "3"],
        rot: Float[np.ndarray, "3 3"],
        childclass: str,
    ) -> None:
        childclass = body.get("childclass", childclass)
        body_pos = coord @ mjcf.parse_vec(body.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
        body_rot = coord @ mjcf.parse_orientation(body) @ coord.T
        name = body.get("name", "")
        if name in by_name:
            # Skeleton bodies start a new joint frame; fixed bodies such as tips fold into their parent.
            owner = by_name[name]
            parsed.local_offsets[owner] = body_pos
            parsed.rest_local_rotations[owner] = body_rot
            pos, rot = np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)
            joint = body.find("joint")
            if joint is not None:
                attrs = {**joint_defaults.get(joint.get("class", childclass), {}), **joint.attrib}
                axis = coord @ mjcf.parse_vec(attrs.get("axis"), default=np.array([0, 0, 1], dtype=np.float32), size=3)
                limit = mjcf.parse_vec(attrs.get("range"), default=np.array([-np.inf, np.inf]), size=2)
                parsed.joint_indices.append(owner)
                parsed.joint_axes.append(axis / np.linalg.norm(axis))
                parsed.joint_limits.append(limit)
        else:
            pos, rot = pos + rot @ body_pos, rot @ body_rot

        for geom in body.findall("geom"):
            merged = ET.Element("geom", {**geom_defaults.get(geom.get("class", childclass), {}), **geom.attrib})
            mesh_name = merged.get("mesh")
            if mesh_name is None:
                continue
            mesh_file = mesh_file_by_name.get(mesh_name)
            if mesh_file is None:
                raise FileNotFoundError(f"Allegro XML references missing mesh asset: {mesh_name}")
            geom_pos = coord @ mjcf.parse_vec(merged.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
            geom_rot = coord @ mjcf.parse_orientation(merged) @ coord.T
            parsed.links.append((owner, mesh_file, pos + rot @ geom_pos, rot @ geom_rot, name))

        for child in body.findall("body"):
            walk(child, owner, pos, rot, childclass)

    walk(palm, 0, np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32), "main")
    if len(parsed.joint_indices) != len(BODY_NAMES) - 1:
        raise ValueError(f"Allegro XML must define {len(BODY_NAMES) - 1} hinge joints")
    return parsed


def _class_defaults(root: ET.Element, tag: str) -> dict[str, dict[str, str]]:
    """Resolve inherited ``<default>`` attributes for one element tag by class name."""
    defaults: dict[str, dict[str, str]] = {}

    def walk(default: ET.Element, inherited: dict[str, str]) -> None:
        element = default.find(tag)
        attrs = {**inherited, **(element.attrib if element is not None else {})}
        defaults[default.get("class", "main")] = attrs
        for child in default.findall("default"):
            walk(child, attrs)

    for default in root.findall("default"):
        walk(default, {})
    return defaults


def _load_link_meshes(
    mesh_dir: Path,
    links: list[tuple[int, str, Float[np.ndarray, "3"], Float[np.ndarray, "3 3"], str]],
    side: str,
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
    for joint_index, mesh_file, geom_pos, geom_rot, body_name in links:
        path = mesh_dir / mesh_file
        if not path.exists():
            raise FileNotFoundError(f"Allegro mesh not found: {path}")
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
        link_data["names"].append(f"{side}_{body_name}")
        vertex_offset += vertices.shape[0]
        face_offset += faces.shape[0]
    if not vertices_by_link:
        raise FileNotFoundError(f"No Allegro STL link meshes found in {mesh_dir}")
    link_data["geom_positions"] = np.asarray(link_data["geom_positions"])
    link_data["geom_rotations"] = np.asarray(link_data["geom_rotations"])
    return np.concatenate(vertices_by_link), np.concatenate(faces_by_link), link_data


def _has_model(path: Path) -> bool:
    return all((path / f"{side}.xml").exists() and (path / "meshes" / side).is_dir() for side in VALID_SIDES)
