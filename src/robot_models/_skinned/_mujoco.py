"""Shape a skinned robot's MuJoCo model specification."""

from pathlib import Path
from typing import Any

import numpy as np
from trimesh import Trimesh

from robot_models._common import mjcf
from robot_models._common.mjcf import MUJOCO_TO_MODEL


def apply_shape(spec: Any, directory: Path, variant: str, shape: Any) -> None:
    """Move a skinned robot's spec to Body Models shape coefficients, before compilation.

    The joint offsets, link and piece meshes move as the rest skin does, and each link's mass,
    center and inertia follow from its shaped mesh at the robot's tissue density.
    """
    rest = mjcf.load(directory / f"{variant}.xml")
    with np.load(directory / f"{variant}.npz") as data:
        dirs = {key: data[key] for key in ("offset_dirs", "visual_dirs", "collision_dirs")}
        density = float(data["density"])
    shape = np.asarray(shape, np.float64)
    if shape.shape != (len(dirs["offset_dirs"]),):
        raise ValueError(f"shape must have {len(dirs['offset_dirs'])} coefficients, got shape {shape.shape}")
    shaped = {key: np.tensordot(shape, value, axes=1) for key, value in dirs.items()}

    def mujoco_frame(vectors: np.ndarray) -> np.ndarray:
        return vectors @ MUJOCO_TO_MODEL  # rows of model (Y-up) vectors to MuJoCo (Z-up)

    offsets = rest.local_offsets + shaped["offset_dirs"]
    for name, parent, offset in zip(rest.joint_names, rest.parents, offsets, strict=True):
        if parent >= 0:
            spec.body(name).pos = mujoco_frame(offset)
    for group, key in ((rest.visual, "visual_dirs"), (rest.collision, "collision_dirs")):
        vertices = group.vertices + shaped[key]
        for index, joint in enumerate(group.joint_indices):
            start, end = group.vertex_offsets[index], group.vertex_offsets[index + 1]
            if group is rest.visual:
                mesh_name = group.names[index]
                faces = group.faces[group.face_offsets[index] : group.face_offsets[index + 1]] - start
                link = Trimesh(vertices[start:end], faces, process=False)
                body = spec.body(rest.joint_names[joint])
                inertia = MUJOCO_TO_MODEL.T @ (link.moment_inertia * density) @ MUJOCO_TO_MODEL
                body.mass = link.volume * density
                body.ipos = mujoco_frame(link.center_mass)
                body.fullinertia = inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
            else:
                mesh_name = spec.geom(group.names[index]).meshname
            spec.mesh(mesh_name).uservert = mujoco_frame(vertices[start:end]).ravel()
