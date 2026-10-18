"""Shape a skinned robot's MuJoCo model specification."""

from pathlib import Path
from typing import Any

from trimesh import Trimesh

from robot_models._common.mjcf import MUJOCO_TO_MODEL
from robot_models._skinned._geometry import geometry


def apply_geometry(spec: Any, name: str, xml_path: Path, shape: Any, conform_to: Any) -> None:
    """Give a skinned robot's spec the geometry of a model of ``shape`` and ``conform_to``, before compilation.

    The joint offsets, link and piece meshes are that model's, and each link's mass, center and
    inertia follow from its mesh at the robot's tissue density.
    """
    shaped = geometry(name, xml_path, shape, conform_to)
    rigid, density = shaped.assets, shaped.density

    def mujoco_frame(vectors):
        return vectors @ MUJOCO_TO_MODEL  # rows of model (Y-up) vectors to MuJoCo (Z-up)

    for joint, parent, offset in zip(rigid.joint_names, rigid.parents, rigid.local_offsets, strict=True):
        if parent >= 0:
            spec.body(joint).pos = mujoco_frame(offset)
    visual = rigid.visual
    for index, joint in enumerate(visual.joint_indices):
        start, end = visual.vertex_offsets[index : index + 2]
        faces = visual.faces[visual.face_offsets[index] : visual.face_offsets[index + 1]] - start
        link = Trimesh(visual.vertices[start:end], faces, process=False)
        inertia = MUJOCO_TO_MODEL.T @ (link.moment_inertia * density) @ MUJOCO_TO_MODEL
        body = spec.body(rigid.joint_names[joint])
        body.mass = link.volume * density
        body.ipos = mujoco_frame(link.center_mass)
        body.fullinertia = inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
        spec.mesh(visual.names[index]).uservert = mujoco_frame(visual.vertices[start:end]).ravel()
    collision = rigid.collision
    for index, geom in enumerate(collision.names):
        start, end = collision.vertex_offsets[index : index + 2]
        spec.mesh(spec.geom(geom).meshname).uservert = mujoco_frame(collision.vertices[start:end]).ravel()
