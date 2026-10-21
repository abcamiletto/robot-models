"""Attach canonical hands through MuJoCo's native model-editing API."""

from importlib import import_module
from typing import Any, cast

from robot_models._common import mjcf
from robot_models._common.mjcf import MUJOCO_TO_MODEL
from robot_models.hands._base import Hand


def add_hand(
    spec: Any,
    model: Hand,
    *,
    frame: Any | None = None,
    prefix: str | None = None,
    free: bool = False,
    actuators: bool = True,
) -> tuple[Any, Any]:
    """Attach a hand model's MJCF and return ``(frame, root_body)``.

    Args:
        spec (mujoco.MjSpec): Scene to modify, before compilation.
        model: The hand, such as ``create_model("shadow", side="left")``: its side, assets and,
            for a hand built from Body Models, its shape and conformed geometry.
        frame (mujoco.MjsFrame | str | None): Existing scene frame, or the world origin.
        prefix: Namespace for attached elements; defaults to ``"{name}_{side}_"``.
        free: Add a free joint at the hand root, before its hinge coordinates.
        actuators: Keep the MJCF's actuators; without them, the hand has none.

    Returns:
        tuple[mujoco.MjsFrame, mujoco.MjsBody]: Mount frame and attached root body.
        Placement uses native MuJoCo coordinates (Z-up, quaternion order wxyz).

    The hand root is fixed unless ``free=True``. Free roots require a world-level mount.
    The scene's options are unchanged. Equalities and collision definitions are retained,
    actuators unless ``actuators=False``; ``effort_limits`` bound every actuator on each coordinate's joint.
    Barrett/Robotiq assets have placeholder inertias for their visual geometry;
    source collisions are supplied; configure appropriate dynamics and actuation.
    Requires the optional ``robot-models[mujoco]`` dependency.
    """
    if not isinstance(model, Hand):
        raise TypeError(f"{model.name!r} is not a hand")
    if isinstance(frame, str):
        found = spec.frame(frame)
        if found is None:
            raise ValueError(f"Frame {frame!r} not found")
        frame = found
    if free and frame is not None and frame.parent != spec.worldbody:
        raise ValueError("free=True requires a world-level mount frame")
    mujoco = cast(Any, import_module("mujoco"))  # Native bindings do not provide type stubs.
    hand = mujoco.MjSpec.from_file(str(model.xml_path))
    if model._config.shape is not None or model._config.conform_to is not None:
        _write_geometry(hand, model)
    if hand.compiler.settotalmass > 0:
        # Global mass normalization is not applied to an attached subtree.
        compiled = hand.compile()
        for body in hand.bodies[1:]:
            index = compiled.body(body.name).id
            body.mass = compiled.body_mass[index]
            body.inertia = compiled.body_inertia[index]
            body.ipos = compiled.body_ipos[index]
            body.iquat = compiled.body_iquat[index]
            body.explicitinertial = True
        hand.compiler.settotalmass = -1
    hand.option = spec.option
    hand.njmax = spec.njmax
    hand.nconmax = spec.nconmax
    (root,) = hand.worldbody.bodies
    root.pos = [0, 0, 0]
    root.quat = [1, 0, 0, 0]
    for joint in list(root.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            hand.delete(joint)
    if free:
        root.add_freejoint()
    if not actuators:
        for actuator in list(hand.actuators):
            hand.delete(actuator)
    for name, effort in zip(model.dof_names, model.runtime.to_numpy(model.effort_limits), strict=True):
        hand.joint(name).actfrcrange = [-effort, effort]
        hand.joint(name).actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
    frame = spec.worldbody.add_frame() if frame is None else frame
    prefix = f"{model.name}_{model.side}_" if prefix is None else prefix
    root_name = prefix + root.name  # attaching prefixes the root's own name
    attachment = spec.attach(hand, frame=frame, prefix=prefix)
    return attachment, spec.body(root_name)


def _write_geometry(spec: Any, model: Hand) -> None:
    """Give a skinned hand's spec the model's geometry: its body offsets, meshes and link inertias."""
    density = model._source.density  # type: ignore[attr-defined]
    offsets = model.runtime.to_numpy(model._assets.local_offsets)
    for body, parent, offset in zip(model.body_names, model.parents, offsets, strict=True):
        if parent >= 0:
            spec.body(body).pos = offset @ MUJOCO_TO_MODEL
    visual = model.visual
    for name, body, mesh in zip(visual.names, visual.body_indices, visual.meshes, strict=True):
        mass, center, full = mjcf.inertial(mesh, density)
        link = spec.body(model.body_names[body])
        link.mass, link.ipos, link.fullinertia = mass, center, full
        spec.mesh(spec.geom(name).meshname).uservert = (mesh.vertices @ MUJOCO_TO_MODEL).ravel()
    for name, mesh in zip(model.collision.names, model.collision.meshes, strict=True):
        spec.mesh(spec.geom(name).meshname).uservert = (mesh.vertices @ MUJOCO_TO_MODEL).ravel()
