"""Attach canonical hand assets through MuJoCo's native model-editing API."""

from importlib import import_module
from math import isfinite
from pathlib import Path
from typing import Any, cast

from robot_models import _assets as assets
from robot_models import _catalog as catalog
from robot_models._spec import Side


def add_hand(
    spec: Any,
    name: str,
    *,
    side: Side = "right",
    frame: Any | None = None,
    prefix: str | None = None,
    model_path: Path | str | None = None,
    free: bool = False,
    shape: Any | None = None,
    conform_to: Any | None = None,
) -> tuple[Any, Any]:
    """Attach a canonical hand and return ``(frame, root_body)``.

    Args:
        spec (mujoco.MjSpec): Scene to modify, before compilation.
        name: Hand factory identifier, such as ``"shadow"`` or ``"allegro"``.
        side: A side supplied by that hand's canonical assets.
        frame (mujoco.MjsFrame | str | None): Existing scene frame, or the world origin.
        prefix: Namespace for attached elements; defaults to ``"{name}_{side}_"``.
        model_path: Optional canonical asset directory, as in ``create_model``.
        free: Add a free joint at the hand root, before its hinge coordinates.
        shape: Body Models shape coefficients of a hand built from Body Models (``"mano"``),
            which move its joints, meshes and link inertias; the mean shape by default.
        conform_to: The hand pose such a hand's links and collision pieces are cut at, as in
            its model's constructor; flat by default.

    Returns:
        tuple[mujoco.MjsFrame, mujoco.MjsBody]: Mount frame and attached root body.
        Placement uses native MuJoCo coordinates (Z-up, quaternion order wxyz).

    The hand root is fixed unless ``free=True``. Free roots require a world-level mount.
    The scene's options are unchanged. Packaged actuators,
    equalities and collision definitions are retained; none are synthesized.
    Barrett/Robotiq assets have placeholder inertias for their visual geometry;
    source collisions are supplied; configure appropriate dynamics and actuation.
    Requires the optional ``robot-models[mujoco]`` dependency.
    """
    name = catalog.normalize_name(name)
    definition = catalog.model_spec(name)
    if definition.family != "hands":
        raise TypeError(f"{name!r} is not a hand")
    if side not in definition.sides:
        raise ValueError(f"{name!r} supports side(s): {definition.sides}")
    if isinstance(frame, str):
        found = spec.frame(frame)
        if found is None:
            raise ValueError(f"Frame {frame!r} not found")
        frame = found
    if free and frame is not None and frame.parent != spec.worldbody:
        raise ValueError("free=True requires a world-level mount frame")
    if (shape is not None or conform_to is not None) and definition.body_model is None:
        raise ValueError(f"{name!r} has no shape or conformed geometry; only hands built from Body Models do")
    mujoco = cast(Any, import_module("mujoco"))  # Native bindings do not provide type stubs.
    directory = assets.model_dir(name, model_path)
    hand = mujoco.MjSpec.from_file(str(directory / f"{side}.xml"))
    if shape is not None or conform_to is not None:
        from robot_models._skinned._mujoco import apply_geometry

        apply_geometry(hand, name, directory / f"{side}.xml", shape, conform_to)
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
    efforts = {name.format(side=side, s=side[0]): dof.effort_limit for name, dof in definition.dofs.items()}
    for actuator in hand.actuators:
        limit = efforts.get(actuator.target, float("inf"))
        gear = abs(actuator.gear[0])
        if actuator.trntype == mujoco.mjtTrn.mjTRN_JOINT and isfinite(limit) and gear:
            force = limit / gear
            lower, upper = -force, force
            if (
                actuator.forcelimited != mujoco.mjtLimited.mjLIMITED_FALSE
                and actuator.forcerange[0] < actuator.forcerange[1]
            ):
                lower = max(lower, actuator.forcerange[0])
                upper = min(upper, actuator.forcerange[1])
            actuator.forcerange = [lower, upper]
            actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    frame = spec.worldbody.add_frame() if frame is None else frame
    prefix = f"{name}_{side}_" if prefix is None else prefix
    root_name = prefix + root.name
    attachment = spec.attach(hand, frame=frame, prefix=prefix)
    return attachment, spec.body(root_name)
