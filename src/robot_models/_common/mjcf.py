"""Load MuJoCo MJCF robots as rigid skeletons with link meshes."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path

import numpy as np
import trimesh
from jaxtyping import Float, Int
from nanomanifold import SO3
from numpy.polynomial import Polynomial

from robot_models._common import stl
from robot_models._common.rigid import MeshAssets, RigidAssets

# Rows map MuJoCo's Z-up axes to the Y-up model frame.
MUJOCO_TO_MODEL = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
IDENTITY_POLYCOEF = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

Defaults = dict[str, dict[str, dict[str, str]]]
Mesh = tuple[Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"]]


def load(xml_path: Path) -> RigidAssets:
    """Load an MJCF robot with one skeleton joint per body, in Y-up model coordinates.

    The root body sits at the origin, as under a MuJoCo free joint at zero. Visual
    meshes are mesh geoms with ``contype`` and ``conaffinity`` set to zero.
    Collision meshes come from contact-enabled geoms and explicit contact pairs.
    Curved primitives are tessellated for this mesh interface. A body's hinges
    compose in MJCF order, as in MuJoCo. A hinge
    named as ``joint1`` of an ``<equality><joint>`` follows ``joint1 = polycoef(joint2)``;
    every other hinge is a pose coordinate, in MuJoCo ``qpos`` order. A coordinate's limits are
    what it can reach in MuJoCo: its enforced range, narrowed to where its coupled hinges stay
    within theirs.
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
    visuals: list[tuple[str, int, Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"]]] = []
    collisions: list[tuple[str, int, Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"]]] = []
    paired = {pair.attrib[key] for pair in root.findall("contact/pair") for key in ("geom1", "geom2")}
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
        for geom_index, geom in enumerate(body.findall("geom")):
            attrs = _attrs(geom, cls, defaults)
            contact = int(attrs.get("contype", "1")) or int(attrs.get("conaffinity", "1"))
            visual = "mesh" in attrs and not contact
            collision = contact or attrs.get("name") in paired
            if not (visual or collision):
                continue
            vertices, faces = _geom_mesh(attrs, meshes)
            if visual:
                visuals.append((attrs["mesh"], index, vertices, faces))
            if collision:
                name = attrs.get("name", f"{body.get('name')}_collision_{geom_index}")
                collisions.append((name, index, vertices, faces))
    coupling = _couplings(root)
    hinge_names = [name for name, *_ in hinges]
    actuated = [name for name in hinge_names if name not in coupling]
    resolved = [coupling.get(name, (name, IDENTITY_POLYCOEF)) for name in hinge_names]
    drivers = [actuated.index(driver) for driver, _ in resolved]
    polycoef = [coef for _, coef in resolved]
    limits = {name: limit for name, _, _, limit in hinges}
    for name, (driver, coef) in coupling.items():
        limits[driver] = _reachable(coef, limits[driver], limits[name])

    to_model = MUJOCO_TO_MODEL
    return RigidAssets(
        joint_names=names,
        parents=parents,
        local_offsets=(np.asarray(offsets) @ to_model.T).astype(np.float32),
        rest_local_rotations=(to_model @ np.asarray(rotations) @ to_model.T).astype(np.float32),
        hinge_names=hinge_names,
        hinge_joint_indices=[joint for _, joint, _, _ in hinges],
        hinge_axes=np.asarray([to_model @ axis for _, _, axis, _ in hinges], dtype=np.float32),
        hinge_drivers=drivers,
        hinge_polycoef=np.asarray(polycoef, dtype=np.float32),
        actuated_joint_names=actuated,
        actuated_joint_limits=np.asarray([limits[name] for name in actuated], dtype=np.float32),
        visual=_mesh_assets(visuals),
        collision=_mesh_assets(collisions),
    )


def _mesh_assets(pieces: list[tuple[str, int, np.ndarray, np.ndarray]]) -> MeshAssets:
    vertex_counts = [len(vertices) for _, _, vertices, _ in pieces]
    face_counts = [len(faces) for *_, faces in pieces]
    vertex_offsets = np.cumsum([0, *vertex_counts]).tolist()
    face_offsets = np.cumsum([0, *face_counts]).tolist()
    vertices = [vertices @ MUJOCO_TO_MODEL.T for _, _, vertices, _ in pieces]
    faces = [faces + start for (*_, faces), start in zip(pieces, vertex_offsets[:-1], strict=True)]
    return MeshAssets(
        vertices=np.concatenate(vertices).astype(np.float32) if pieces else np.empty((0, 3), dtype=np.float32),
        faces=np.concatenate(faces).astype(np.int64) if pieces else np.empty((0, 3), dtype=np.int64),
        names=[name for name, *_ in pieces],
        joint_indices=[joint for _, joint, _, _ in pieces],
        vertex_offsets=vertex_offsets,
        face_offsets=face_offsets,
    )


def _geom_mesh(attrs: dict[str, str], meshes: dict[str, Mesh]) -> Mesh:
    """Triangulate a source geom, retaining its body-local pose and dimensions."""
    kind = attrs.get("type", "mesh" if "mesh" in attrs else "sphere")
    rotation = _orientation(attrs)
    position = _vec(attrs.get("pos"), 3)
    size = _vec(attrs.get("size", "0 0 0"))
    if "fromto" in attrs:
        start, end = _vec(attrs["fromto"], 6).reshape(2, 3)
        direction = end - start
        half_length = np.linalg.norm(direction) / 2
        rotation = _zaxis_rotation(direction)
        position = (start + end) / 2
        size = (
            np.array([size[0], half_length])
            if kind in ("capsule", "cylinder")
            else np.array([size[0], size[0], half_length])
        )
    if kind == "mesh":
        vertices, faces = meshes[attrs["mesh"]]
    else:
        if kind == "box":
            mesh = trimesh.creation.box(extents=2 * size)
        elif kind == "sphere":
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
        elif kind == "ellipsoid":
            mesh = trimesh.creation.icosphere(subdivisions=2)
            mesh.vertices *= size
        elif kind == "cylinder":
            mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=32)
        elif kind == "capsule":
            mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1])
            mesh.vertices -= mesh.bounds.mean(axis=0)
        else:
            raise ValueError(f"Unsupported collision geometry {kind!r}")
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    return vertices @ rotation.T + position, faces


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
    """Return a hinge's unit axis and enforced range."""
    if attrs.get("type", "hinge") != "hinge" or np.any(_vec(attrs.get("pos"), 3)):
        raise ValueError(f"Joint {attrs['name']!r} must be a hinge through its body origin")
    axis = _vec(attrs.get("axis", "0 0 1"), 3)
    enforced = "range" in attrs and attrs.get("limited") != "false"
    lo, hi = _vec(attrs["range"], 2) if enforced else (-np.inf, np.inf)
    return axis / np.linalg.norm(axis), (float(lo), float(hi))


def _reachable(
    coef: Float[np.ndarray, "5"], driver: tuple[float, float], coupled: tuple[float, float]
) -> tuple[float, float]:
    """The part of a driver's range where its coupled hinge, at angle ``coef(driver)``, stays within ``coupled``."""
    lo, hi = driver
    if lo >= hi or not np.isfinite(coupled).any():
        return driver
    angle = Polynomial(coef)
    roots = [root.real for bound in coupled if np.isfinite(bound) for root in (angle - bound).roots() if not root.imag]
    edges = sorted({lo, hi, *(root for root in roots if lo < root < hi)})
    inside = []
    for a, b in pairwise(edges):
        if coupled[0] <= angle(_middle(a, b)) <= coupled[1]:
            inside.append((a, b))
    if not inside or any(before[1] != after[0] for before, after in pairwise(inside)):
        raise ValueError(f"A coupled hinge's range {coupled} leaves no single interval of its driver's {driver}")
    return inside[0][0], inside[-1][1]


def _middle(a: float, b: float) -> float:
    """The middle of the interval from ``a`` to ``b``, an infinite end replaced by the other end -/+ 2."""
    start = a if np.isfinite(a) else (b - 2 if np.isfinite(b) else -1.0)
    end = b if np.isfinite(b) else start + 2
    return (start + end) / 2


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
    """Load every ``<asset><mesh>`` by name, from its STL file or its inline vertices and faces, scaled."""
    meshes = {}
    for mesh in root.findall("asset/mesh"):
        attrs = _attrs(mesh, "main", defaults)
        scale = _vec(attrs.get("scale", "1 1 1"), 3)
        if "vertex" in attrs:
            vertices = np.fromstring(attrs["vertex"], sep=" ").reshape(-1, 3)
            faces = np.fromstring(attrs["face"], dtype=np.int64, sep=" ").reshape(-1, 3)
            meshes[attrs["name"]] = (vertices * scale, faces[:, ::-1] if np.prod(scale) < 0 else faces)
            continue
        file = Path(attrs["file"])
        meshes[attrs.get("name", file.stem)] = stl.load_stl_mesh(mesh_dir / file, scale=scale)
    return meshes


def _orientation(attrs: dict[str, str]) -> Float[np.ndarray, "3 3"]:
    """Parse MJCF orientation attributes with radian angles and default intrinsic XYZ Euler order."""
    if "quat" in attrs:
        return SO3.conversions.from_quat_to_rotmat(_vec(attrs["quat"], 4), convention="wxyz", xp=np)
    if "euler" in attrs:
        return SO3.conversions.from_euler_to_rotmat(_vec(attrs["euler"], 3), convention="XYZ", xp=np)
    if "axisangle" in attrs:
        axis_angle = _vec(attrs["axisangle"], 4)
        rotvec = axis_angle[:3] * axis_angle[3] / np.linalg.norm(axis_angle[:3])
        return SO3.convert(rotvec, src="axis_angle", dst="rotmat", xp=np)
    if "zaxis" in attrs:
        return _zaxis_rotation(_vec(attrs["zaxis"], 3))
    if "xyaxes" in attrs:
        x, y = _vec(attrs["xyaxes"], 6).reshape(2, 3)
        x = x / np.linalg.norm(x)
        y = y - np.dot(x, y) * x
        y = y / np.linalg.norm(y)
        return np.column_stack((x, y, np.cross(x, y)))
    return np.eye(3)


def _zaxis_rotation(direction: np.ndarray) -> np.ndarray:
    x, y, z = direction / np.linalg.norm(direction)
    quat = np.array([1 + z, -y, x, 0]) if z > -1 + 1e-12 else np.array([0.0, 1.0, 0.0, 0.0])
    quat /= np.linalg.norm(quat)
    return SO3.conversions.from_quat_to_rotmat(quat, convention="wxyz", xp=np)


def _vec(value: str | None, size: int | None = None) -> Float[np.ndarray, "N"]:
    vector = np.zeros(size or 0) if value is None else np.asarray([float(x) for x in value.split()])
    if size is not None and vector.shape != (size,):
        raise ValueError(f"Expected {size} values, got {value!r}")
    return vector
