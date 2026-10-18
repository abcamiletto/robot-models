"""A skinned robot's rigid description: the Body Models skin at a shape and pose, cut per joint."""

import functools
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from nanomanifold import SO3

from robot_models._common import mjcf
from robot_models._common import rigid as rigid_ops
from robot_models._common.rigid import RigidAssets
from robot_models._skinned._sources import numpy_model, source_of


@dataclass(frozen=True)
class Geometry:
    """A variant's rigid assets, its root joint's rest position in Body Models' frame, its shape
    coefficients, tissue density (kg/m³) and each moving joint's Body Models pose row."""

    assets: RigidAssets
    root: np.ndarray
    shape: np.ndarray
    density: float
    pose_rows: list[int]


def geometry(name: str, xml_path: Path, shape: Sequence[float] | None, conform_to: Sequence[float] | None) -> Geometry:
    """The built variant at ``shape`` (the mean by default), its links and pieces conformed to the skin at
    ``conform_to``, this model's hinge angles (flat by default).

    The joints are Body Models' rest joints at the shape. Each link is the skin at the shape and pose, cut
    as at build and expressed in its joint's posed frame; each piece vertex follows its nearest link vertices.
    """
    source, body = source_of(name), numpy_model(name, xml_path.stem)
    rest, data = _rest(xml_path, xml_path.stat().st_mtime_ns)
    count = body.parameter_spec["shape"].dims[0]
    coefficients = np.zeros(count) if shape is None else np.asarray(shape, np.float64)
    pose = np.zeros(len(rest.actuated_joint_names)) if conform_to is None else np.asarray(conform_to, np.float64)

    params = body.get_rest_pose(dtype=np.float64) | {"shape": coefficients}
    joints = np.asarray(body.forward_skeleton(**params))[[body.joint_names.index(n) for n in rest.joint_names], :3, 3]
    offsets = joints - joints[np.maximum(rest.parents, 0)]
    offsets[np.asarray(rest.parents) < 0] = 0.0
    assets = replace(rest, local_offsets=offsets)

    frames = rigid_ops.forward_skeleton(
        assets, pose, global_rotation=None, global_translation=None, joint_indices=None, xp=np
    )
    local = np.einsum("jki,jkl->jil", frames[np.asarray(rest.parents)[1:], :3, :3], frames[1:, :3, :3])
    body_pose = np.zeros((len(local), 3))
    body_pose[data["pose_rows"]] = SO3.convert(local, src="rotmat", dst="axis_angle", xp=np)
    skin = np.asarray(body.forward_vertices(**params | {source.pose: body_pose})) - joints[0]

    joint = np.repeat(rest.visual.joint_indices, np.diff(rest.visual.vertex_offsets))
    world = (data["skin_weight"][..., None] * skin[data["skin_index"]]).sum(1)
    visual = np.einsum("vki,vk->vi", frames[joint, :3, :3], world - frames[joint, :3, 3])
    moved = visual - rest.visual.vertices
    collision = rest.collision.vertices + (data["piece_weight"][..., None] * moved[data["piece_index"]]).sum(1)
    assets = replace(
        assets,
        visual=replace(rest.visual, vertices=visual.astype(np.float32)),
        collision=replace(rest.collision, vertices=collision.astype(np.float32)),
        local_offsets=offsets.astype(np.float32),
    )
    root, density = joints[0].astype(np.float32), float(data["density"])
    return Geometry(assets, root, coefficients.astype(np.float32), density, data["pose_rows"].tolist())


@functools.cache
def _rest(xml_path: Path, modified: int) -> tuple[RigidAssets, dict[str, np.ndarray]]:
    """A built variant's flat, mean-shape assets and its build data, parsed once per file version."""
    with np.load(xml_path.with_suffix(".npz")) as data:
        return mjcf.load(xml_path), {key: data[key] for key in data.files}
