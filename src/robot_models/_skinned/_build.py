"""Build a skinned robot's assets from the user's own Body Models files, into a local directory.

The links are the rest skin cut by joint: each face goes to the joint that skins its corners
most, and each link is closed by a fan over every opening. The collision pieces are a CoACD
convex decomposition (Wei et al., SIGGRAPH 2022) of the base variant's mean-shape links; other
variants and shapes carry each piece vertex with its link's skin, by inverse squared distance
over its ``NEAREST`` rest link vertices. Every rest quantity is linear in the shape
coefficients, so each variant stores its mean-shape geometry in its MJCF and the shape
directions of its joint offsets and vertices in ``{variant}.npz``. Nothing built here may be
redistributed: it is derived from the MANO or SMPL model files.
"""

from __future__ import annotations

import json
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh

from robot_models._common.mjcf import MUJOCO_TO_MODEL
from robot_models._skinned._sources import MIRROR, SOURCES, Source

ASSET_VERSION = 2
NEAREST = 4  # rest link vertices each piece vertex follows
STIFFNESS = 10.0  # N·m/rad of the packaged position actuators
DAMPING = 0.003  # N·m·s/rad per hinge
ARMATURE = 1e-4  # kg·m²
CONTACT_MARGIN = 0.002  # m: link pairs this close at rest do not collide with each other


def ensure(name: str, directory: Path) -> Path:
    """Return ``directory`` holding current assets of every configured variant, building them if needed."""
    source = SOURCES[name]
    sources = {}
    for variant in source.variants:
        try:
            sources[variant] = source.path(variant)
        except FileNotFoundError:
            continue
    if source.base not in sources:
        raise FileNotFoundError(
            f"{name} is built locally from Body Models' {source.body_model.upper()} files, which cannot be "
            f"redistributed, starting from its {source.base} model. Configure it with "
            f"`body-models set {source.body_model}-{source.base} PATH`."
        )
    manifest = _manifest(directory)
    built = manifest.get("variants", {})
    fresh = all(built.get(variant, {}).get("stamp") == _stamp(path) for variant, path in sources.items())
    if manifest.get("asset_version") == ASSET_VERSION and fresh:
        return directory
    build(name, directory, sources)
    return directory


def build(name: str, directory: Path, sources: dict[str, Path]) -> None:
    """Build every variant in ``sources`` into ``directory``, replacing it once complete."""
    source = SOURCES[name]
    print(f"Building {name} from Body Models' local {source.body_model.upper()} files into {directory}...")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{directory.name}-", dir=directory.parent) as temporary:
        staging = Path(temporary) / "assets"
        staging.mkdir()
        pieces = _pieces(source, directory, staging)
        variants = {}
        for variant, path in sources.items():
            _write_variant(source, variant, pieces, staging)
            variants[variant] = {"source": str(path), "stamp": _stamp(path)}
        manifest = {"asset_version": ASSET_VERSION, "body_model": source.body_model, "variants": variants}
        (staging / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
        previous = Path(temporary) / "previous"
        if directory.exists():
            directory.rename(previous)
        try:
            staging.rename(directory)
        except OSError:
            if previous.exists():
                previous.rename(directory)
            raise


@dataclass(frozen=True)
class Skin:
    """A variant's rest skin and joints, linear in shape: ``x(shape) = x + shape @ x_dirs``."""

    names: list[str]
    parents: list[int]
    faces: np.ndarray
    weights: np.ndarray
    vertices: np.ndarray  # (V, 3)
    vertex_dirs: np.ndarray  # (S, V, 3)
    joints: np.ndarray  # (J, 3)
    joint_dirs: np.ndarray  # (S, J, 3)


@dataclass(frozen=True)
class Cut:
    """One link's vertices as skin vertex indices, then one fan center per opening."""

    faces: np.ndarray  # into the link's vertices
    used: np.ndarray  # skin vertex indices
    loops: list[np.ndarray]  # positions in ``used`` of each opening's rim

    def apply(self, skin: np.ndarray) -> np.ndarray:
        """The link's vertices for skin vertices ``[..., V, 3]``."""
        own = skin[..., self.used, :]
        centers = [own[..., loop, :].mean(-2, keepdims=True) for loop in self.loops]
        return np.concatenate([own, *centers], axis=-2)


def skin(source: Source, variant: str) -> Skin:
    """A variant's flat rest skin and joints at the mean shape, and their shape directions."""
    model = source.create(variant)
    shapes = model.parameter_spec["shape"].dims[0]
    params = {k: np.asarray(v) for k, v in model.get_rest_pose(batch_dims=(shapes + 1,), dtype=np.float64).items()}
    params["shape"] = np.concatenate([np.zeros((1, shapes)), np.eye(shapes)]).astype(params["shape"].dtype)
    vertices = np.asarray(model.forward_vertices(**params), np.float64)
    joints = np.asarray(model.forward_skeleton(**params), np.float64)[..., :3, 3]
    return Skin(
        names=list(model.joint_names),
        parents=[int(p) for p in model.parents],
        faces=np.asarray(model.faces, np.int64),
        weights=np.asarray(model.skin_weights, np.float64),
        vertices=vertices[0],
        vertex_dirs=vertices[1:] - vertices[0],
        joints=joints[0],
        joint_dirs=joints[1:] - joints[0],
    )


def cut(skin: Skin) -> list[Cut]:
    """Each joint's link: the faces it skins most, each opening closed by a fan to its centroid."""
    owner = skin.weights[skin.faces].sum(1).argmax(1)
    cuts = []
    for joint in range(len(skin.names)):
        faces = skin.faces[owner == joint]
        used = np.unique(faces)
        remap = np.full(len(skin.vertices), -1)
        remap[used] = np.arange(len(used))
        local = remap[faces]
        edges = np.concatenate([local[:, [0, 1]], local[:, [1, 2]], local[:, [2, 0]]])
        _, inverse, counts = np.unique(np.sort(edges, 1), axis=0, return_inverse=True, return_counts=True)
        following: dict[int, list[int]] = {}
        for a, b in edges[counts[inverse.ravel()] == 1]:
            following.setdefault(int(a), []).append(int(b))
        loops, caps = [], []
        while following:  # walk each rim; a vertex where two rims touch has two successors
            start = next(iter(following))
            loop = [start]
            while True:
                after = following[loop[-1]].pop()
                if not following[loop[-1]]:
                    del following[loop[-1]]
                if after == start:
                    break
                loop.append(after)
            center = len(used) + len(loops)
            caps += [[b, a, center] for a, b in zip(loop, loop[1:] + loop[:1])]
            loops.append(np.array(loop))
        cuts.append(Cut(np.concatenate([local, np.array(caps, np.int64).reshape(-1, 3)]), used, loops))
    return cuts


def _pieces(source: Source, previous: Path, staging: Path) -> dict[str, np.ndarray]:
    """The base variant's convex pieces (hull vertices in their link's rest frame), reused when current."""
    path = source.path(source.base)
    cached = previous / "pieces.npz"
    if cached.is_file():
        with np.load(cached) as data:
            pieces: dict[str, Any] = dict(data)
        if str(pieces.pop("stamp")) == _stamp(path):
            np.savez_compressed(staging / "pieces.npz", **pieces, stamp=np.array(_stamp(path)))
            return pieces
    import coacd

    coacd.set_log_level("error")
    base = skin(source, source.base)
    vertices, piece, link = [], [], []
    for joint, link_cut in enumerate(cut(base)):
        local = link_cut.apply(base.vertices) - base.joints[joint]
        parts = coacd.run_coacd(
            coacd.Mesh(local, link_cut.faces),
            threshold=source.concavity(base.names[joint]),
            max_convex_hull=-1,
            preprocess_mode="off",
            mcts_iterations=150,
            seed=0,
        )
        for part_vertices, part_faces in parts:
            hull = trimesh.Trimesh(part_vertices, part_faces).convex_hull
            if hull.volume > 1e-12:
                vertices.append(np.asarray(hull.vertices))
                piece.append(np.full(len(hull.vertices), len(link)))
                link.append(joint)
    pieces = {
        "vertices": np.concatenate(vertices),
        "piece": np.concatenate(piece).astype(np.int64),
        "link": np.array(link, np.int64),
    }
    np.savez_compressed(staging / "pieces.npz", **pieces, stamp=np.array(_stamp(path)))
    return pieces


def _write_variant(source: Source, variant: str, pieces: dict[str, np.ndarray], staging: Path) -> None:
    own = skin(source, variant)
    base = own if variant == source.base else skin(source, source.base)
    own_cuts, base_cuts = cut(own), cut(base)
    sign = MIRROR if variant in source.mirrored else np.eye(3)
    order = _depth_first(own.parents)
    moving = [name for name, parent in zip(own.names, own.parents) if parent >= 0]
    hinges = dict(zip(moving, source.hinges(variant, own.joints, own.parents, own.names), strict=True))

    # Links and their shape directions, in their joint's rest frame.
    links = [own_cuts[j].apply(own.vertices) - own.joints[j] for j in range(len(own.names))]
    link_dirs = [own_cuts[j].apply(own.vertex_dirs) - own.joint_dirs[:, None, j] for j in range(len(own.names))]

    # Pieces carried from the (mirrored) base links to this variant's skin, by the base cut's vertices.
    split = np.split(pieces["vertices"], np.flatnonzero(np.diff(pieces["piece"])) + 1)
    piece_vertices: list[list[np.ndarray]] = [[] for _ in own.names]
    piece_dirs: list[list[np.ndarray]] = [[] for _ in own.names]
    for vertices, joint in zip(split, pieces["link"], strict=True):
        link_cut = base_cuts[joint]
        rest = link_cut.apply(base.vertices) @ sign
        world = (vertices + base.joints[joint]) @ sign
        distance, nearest = _nearest(rest, world)
        weights = 1.0 / np.maximum(distance, 1e-6) ** 2
        weights /= weights.sum(1, keepdims=True)
        carried = world + _follow(weights, nearest, link_cut.apply(own.vertices) - rest)
        piece_vertices[joint].append(carried - own.joints[joint])
        moved = _follow(weights, nearest, link_cut.apply(own.vertex_dirs))
        piece_dirs[joint].append(moved - own.joint_dirs[:, None, joint])

    root = order[0]
    offsets = [np.zeros(3) if own.parents[j] < 0 else own.joints[j] - own.joints[own.parents[j]] for j in order]
    parent_dirs = [own.joint_dirs[:, j if own.parents[j] < 0 else own.parents[j]] for j in order]
    offset_dirs = np.stack([own.joint_dirs[:, j] - parent for j, parent in zip(order, parent_dirs)], axis=1)
    np.savez_compressed(
        staging / f"{variant}.npz",
        pose_rows=np.array([joint - 1 for joint in order[1:]], np.int64),
        offset_dirs=offset_dirs,
        visual_dirs=np.concatenate([link_dirs[j] for j in order], axis=1),
        collision_dirs=np.concatenate([d for j in order for d in piece_dirs[j]], axis=1),
        root=own.joints[root],
        root_dirs=own.joint_dirs[:, root],
        density=np.float64(source.density),
    )
    xml = _mjcf(source, variant, own, order, offsets, hinges, links, own_cuts, piece_vertices)
    (staging / f"{variant}.xml").write_text(xml)
    _exclude_touching(staging / f"{variant}.xml")


def _follow(weights: np.ndarray, nearest: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Each point's weighted average of its nearest link vertices' values ``x[..., V, 3]``."""
    return (weights[..., None] * x[..., nearest, :]).sum(-2)


def _nearest(points: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(queries[:, None] - points[None], axis=-1)
    nearest = np.argsort(distance, axis=1)[:, :NEAREST]
    return np.take_along_axis(distance, nearest, 1), nearest


def _depth_first(parents: list[int]) -> list[int]:
    order = []

    def visit(parent: int) -> None:
        for child, p in enumerate(parents):
            if p == parent:
                order.append(child)
                visit(child)

    visit(-1)
    return order


def _mjcf(source, variant, own, order, offsets, hinges, links, cuts, pieces) -> str:
    to_mujoco = MUJOCO_TO_MODEL.T  # model (Y-up) vectors to MuJoCo (Z-up)
    root = ET.Element("mujoco", model=f"{source.body_model}_{variant}")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false", autolimits="true")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping=_fmt(DAMPING), armature=_fmt(ARMATURE), limited="false")
    visual = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(visual, "geom", type="mesh", contype="0", conaffinity="0", group="2", rgba="0.8 0.65 0.55 1")
    collision = ET.SubElement(default, "default", {"class": "collision"})
    ET.SubElement(collision, "geom", type="mesh", group="3", rgba="0.5 0.5 0.8 1")
    asset = ET.SubElement(root, "asset")
    bodies = {-1: ET.SubElement(root, "worldbody")}
    actuators = []
    for joint, offset in zip(order, offsets):
        name = own.names[joint]
        link = trimesh.Trimesh(links[joint], cuts[joint].faces, process=False)
        _mesh(asset, name, links[joint] @ to_mujoco.T, cuts[joint].faces)
        body = ET.SubElement(bodies[own.parents[joint]], "body", name=name, pos=_fmt(offset @ to_mujoco.T))
        bodies[joint] = body
        mass = link.volume * source.density
        inertia = to_mujoco @ (link.moment_inertia * source.density) @ to_mujoco.T
        center = _fmt(link.center_mass @ to_mujoco.T)
        full = _fmt(inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]])
        ET.SubElement(body, "inertial", pos=center, mass=_fmt(mass), fullinertia=full)
        if own.parents[joint] < 0 and source.free:
            ET.SubElement(body, "freejoint", name="root")
        if name in hinges:
            axes, names, ranges = hinges[name]
            for index, (axis, hinge) in enumerate(zip(axes, names)):
                attrs = {"name": hinge, "axis": _fmt(axis @ to_mujoco.T)}
                if ranges is not None:
                    attrs["range"] = _fmt(ranges[index])
                ET.SubElement(body, "joint", attrs)
                actuators.append(hinge)
        ET.SubElement(body, "geom", {"class": "visual", "name": f"{name}_visual", "mesh": name})
        for index, vertices in enumerate(pieces[joint]):
            mesh = f"{name}_piece{index}"
            _mesh(asset, mesh, vertices @ to_mujoco.T, hull_faces(vertices))
            ET.SubElement(body, "geom", {"class": "collision", "name": f"{name}_collision{index}", "mesh": mesh})
    if source.actuated:
        actuator = ET.SubElement(root, "actuator")
        for hinge in actuators:
            ET.SubElement(actuator, "position", name=hinge, joint=hinge, kp=_fmt(STIFFNESS))
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


def hull_faces(vertices: np.ndarray) -> np.ndarray:
    """Outward faces of the vertices' convex hull, indexing the vertices as given."""
    from scipy.spatial import ConvexHull

    faces = ConvexHull(vertices).simplices
    corners = vertices[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    inward = ((corners[:, 0] - vertices.mean(0)) * normals).sum(1) < 0
    faces[inward] = faces[inward, ::-1]
    return faces


def _mesh(asset: ET.Element, name: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    ET.SubElement(asset, "mesh", name=name, vertex=_fmt(vertices.ravel()), face=" ".join(map(str, faces.ravel())))


def _exclude_touching(path: Path) -> None:
    """Exclude contacts between links that touch, or nearly, at rest: they share tissue."""
    mujoco = cast(Any, import_module("mujoco"))  # Native bindings do not provide type stubs.
    model = mujoco.MjModel.from_xml_path(str(path))
    model.geom_margin[:] = CONTACT_MARGIN
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pairs = set()
    for contact in data.contact[: data.ncon]:
        a, b = sorted(int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
        if model.body_parentid[b] != a:  # MuJoCo filters parent-child contacts itself
            pairs.add((a, b))
    if not pairs:
        return
    root = ET.parse(path).getroot()
    contact = ET.SubElement(root, "contact")
    for a, b in sorted(pairs):
        ET.SubElement(contact, "exclude", body1=model.body(a).name, body2=model.body(b).name)
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")


def _fmt(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in np.ravel(values))


def _manifest(directory: Path) -> dict:
    path = directory / "source.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _stamp(path: Path) -> str:
    status = path.stat()
    return f"{path}:{status.st_size}:{status.st_mtime_ns}"


__all__ = ["ASSET_VERSION", "build", "cut", "ensure", "skin"]
