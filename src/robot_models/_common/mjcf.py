"""Load MuJoCo MJCF robots as rigid skeletons with link meshes."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3
from numpy.polynomial import Polynomial

from robot_models._common import stl
from robot_models._common.rigid import RigidAssets

# Rows map MuJoCo's Z-up axes to the Y-up model frame.
MUJOCO_TO_MODEL = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
IDENTITY_POLYCOEF = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

Defaults = dict[str, dict[str, dict[str, str]]]
Mesh = tuple[Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"]]


def load(xml_path: Path) -> RigidAssets:
    """Load an MJCF robot with one skeleton joint per body, in Y-up model coordinates.

    The root body sits at the origin, as under a MuJoCo free joint at zero. Visual
    meshes are mesh geoms with ``contype`` and ``conaffinity`` set to zero. A hinge
    named as ``joint1`` of an ``<equality><joint>`` follows ``joint1 = polycoef(joint2)``;
    every other hinge is a pose coordinate, in MuJoCo ``qpos`` order.
    """
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    worldbody = root.find("worldbody")
    if compiler is None or compiler.get("angle") != "radian" or worldbody is None:
        raise ValueError(f"{xml_path} must have <compiler angle='radian'> and a <worldbody>")
    defaults = _defaults(root)
    meshes = _meshes(root, xml_path.parent / compiler.get("meshdir", ""), defaults)

    names, parents, offsets, rotations = [], [], [], []
    hinges: list[tuple[str, int, Float[np.ndarray, "3"], tuple[float, float]]] = []
    links: list[tuple[str, int, Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"]]] = []
    for index, (body, parent, cls) in enumerate(_bodies(worldbody)):
        is_root = parent < 0
        names.append(body.get("name"))
        parents.append(parent)
        offsets.append(np.zeros(3) if is_root else _vec(body.get("pos"), 3))
        rotations.append(np.eye(3) if is_root else _orientation(body.attrib))
        for joint in body.findall("joint"):
            attrs = _attrs(joint, cls, defaults)
            if attrs.get("type") != "free":
                hinges.append((attrs["name"], index, *_hinge(attrs)))
        for geom in body.findall("geom"):
            attrs = _attrs(geom, cls, defaults)
            if "mesh" in attrs and attrs.get("contype") == "0" and attrs.get("conaffinity") == "0":
                vertices, faces = meshes[attrs["mesh"]]
                body_vertices = vertices @ _orientation(attrs).T + _vec(attrs.get("pos"), 3)
                links.append((attrs["mesh"], index, body_vertices, faces))
    if len({joint for _, joint, _, _ in hinges}) != len(hinges):
        raise ValueError(f"{xml_path} has a body with more than one hinge")

    coupling = _couplings(root)
    hinge_names = [name for name, *_ in hinges]
    actuated = [name for name in hinge_names if name not in coupling]
    resolved = [coupling.get(name, (name, IDENTITY_POLYCOEF)) for name in hinge_names]
    drivers = [actuated.index(driver) for driver, _ in resolved]
    polycoef = [coef for _, coef in resolved]
    limits = {name: limit for name, _, _, limit in hinges}

    to_model = MUJOCO_TO_MODEL
    vertex_counts = [len(vertices) for _, _, vertices, _ in links]
    face_counts = [len(faces) for *_, faces in links]
    vertex_starts = np.cumsum([0, *vertex_counts[:-1]]).tolist()
    return RigidAssets(
        joint_names=names,
        parents=parents,
        local_offsets=(np.asarray(offsets) @ to_model.T).astype(np.float32),
        rest_local_rotations=(to_model @ np.asarray(rotations) @ to_model.T).astype(np.float32),
        hinge_joint_indices=[joint for _, joint, _, _ in hinges],
        hinge_axes=np.asarray([to_model @ axis for _, _, axis, _ in hinges], dtype=np.float32),
        hinge_drivers=drivers,
        hinge_polycoef=np.asarray(polycoef, dtype=np.float32),
        actuated_joint_names=actuated,
        actuated_joint_limits=np.asarray([limits[name] for name in actuated], dtype=np.float32),
        vertices=np.concatenate([vertices @ to_model.T for _, _, vertices, _ in links]).astype(np.float32),
        faces=np.concatenate([faces + start for (*_, faces), start in zip(links, vertex_starts)]).astype(np.int64),
        link_names=[name for name, *_ in links],
        link_joint_indices=[joint for _, joint, _, _ in links],
        link_vertex_starts=vertex_starts,
        link_vertex_counts=vertex_counts,
        link_face_starts=np.cumsum([0, *face_counts[:-1]]).tolist(),
        link_face_counts=face_counts,
    )


def _defaults(root: ET.Element) -> Defaults:
    """Map each default class to its element attributes, with inherited values applied."""
    classes: Defaults = {}

    def visit(default: ET.Element, inherited: dict[str, dict[str, str]]) -> None:
        own = {tag: dict(attrs) for tag, attrs in inherited.items()}
        for element in default:
            if element.tag != "default":
                own[element.tag] = own.get(element.tag, {}) | element.attrib
        classes[default.get("class", "main")] = own
        for child in default.findall("default"):
            visit(child, own)

    for default in root.findall("default"):
        visit(default, {})
    return classes


def _attrs(element: ET.Element, cls: str, defaults: Defaults) -> dict[str, str]:
    """Resolve an element's attributes against its default class."""
    return defaults.get(element.get("class", cls), {}).get(element.tag, {}) | element.attrib


def _bodies(worldbody: ET.Element) -> list[tuple[ET.Element, int, str]]:
    """List bodies depth-first, matching MuJoCo body order, with parent index and default class."""
    bodies = []

    def visit(element: ET.Element, parent: int, childclass: str) -> None:
        for body in element.findall("body"):
            cls = body.get("childclass", childclass)
            bodies.append((body, parent, cls))
            visit(body, len(bodies) - 1, cls)

    visit(worldbody, -1, "main")
    return bodies


def _hinge(attrs: dict[str, str]) -> tuple[Float[np.ndarray, "3"], tuple[float, float]]:
    """Return a hinge's unit axis and range."""
    if attrs.get("type", "hinge") != "hinge" or np.any(_vec(attrs.get("pos"), 3)):
        raise ValueError(f"Joint {attrs['name']!r} must be a hinge through its body origin")
    axis = _vec(attrs.get("axis", "0 0 1"), 3)
    lo, hi = (float(value) for value in attrs.get("range", "-inf inf").split())
    return axis / np.linalg.norm(axis), (lo, hi)


def _couplings(root: ET.Element) -> dict[str, tuple[str, Float[np.ndarray, "5"]]]:
    """Map each coupled hinge to its independent driver hinge and five polynomial coefficients.

    A chain such as ``a = f(b), b = g(c)`` resolves to ``a = f(g(c))``.
    """
    direct = {}
    for equality in root.findall("equality/joint"):
        coef = Polynomial(_vec(equality.get("polycoef", "0 1 0 0 0"))).trim()
        direct[equality.get("joint1")] = (equality.get("joint2"), coef)

    def resolve(name: str) -> tuple[str, Polynomial]:
        driver, coef = direct[name]
        if driver not in direct:
            return driver, coef
        root_driver, inner = resolve(driver)
        return root_driver, coef(inner)

    coupling = {}
    for name in direct:
        driver, coef = resolve(name)
        if len(coef.coef) > 5:
            raise ValueError(f"Coupling of {name!r} composes to a polynomial above degree 4")
        coupling[name] = (driver, np.pad(coef.coef, (0, 5 - len(coef.coef))))
    return coupling


def _meshes(root: ET.Element, mesh_dir: Path, defaults: Defaults) -> dict[str, Mesh]:
    """Load every ``<asset><mesh>`` by name, with its scale applied."""
    meshes = {}
    for mesh in root.findall("asset/mesh"):
        attrs = _attrs(mesh, "main", defaults)
        file = Path(attrs["file"])
        scale = _vec(attrs.get("scale", "1 1 1"), 3)
        meshes[attrs.get("name", file.stem)] = stl.load_stl_mesh(mesh_dir / file, scale=scale)
    return meshes


def _orientation(attrs: dict[str, str]) -> Float[np.ndarray, "3 3"]:
    """Parse an MJCF ``quat`` or ``euler`` (MuJoCo's default intrinsic XYZ) as a rotation matrix."""
    if {"axisangle", "xyaxes", "zaxis"} & attrs.keys():
        raise ValueError(f"Unsupported MJCF orientation in {attrs}")
    if "quat" in attrs:
        return SO3.conversions.from_quat_to_rotmat(_vec(attrs["quat"], 4), convention="wxyz", xp=np)
    if "euler" in attrs:
        return SO3.conversions.from_euler_to_rotmat(_vec(attrs["euler"], 3), convention="XYZ", xp=np)
    return np.eye(3)


def _vec(value: str | None, size: int | None = None) -> Float[np.ndarray, "N"]:
    vector = np.zeros(size or 0) if value is None else np.asarray([float(x) for x in value.split()])
    if size is not None and vector.shape != (size,):
        raise ValueError(f"Expected {size} values, got {value!r}")
    return vector
