"""Attach canonical hand assets through MuJoCo's native model-editing API."""

from importlib import import_module
from pathlib import Path
from typing import Any, cast

from robot_models import _assets as assets
from robot_models import _registry as registry
from robot_models._base import Hand, Side


def add_hand(
    spec: Any,
    name: str,
    *,
    side: Side = "right",
    frame: Any | None = None,
    prefix: str | None = None,
    model_path: Path | str | None = None,
) -> Any:
    """Attach one canonical hand to a MuJoCo scene and return its attachment frame.

    Args:
        spec (mujoco.MjSpec): Scene to modify, before compilation.
        name: Hand factory identifier, such as ``"shadow"`` or ``"allegro"``.
        side: A side supplied by that hand's canonical assets.
        frame (mujoco.MjsFrame | str | None): Existing scene frame, or the world origin.
        prefix: Namespace for attached elements; defaults to ``"{name}_{side}_"``.
        model_path: Optional canonical asset directory, as in ``create_model``.

    Returns:
        mujoco.MjsFrame: Attachment frame; its ``pos`` and ``quat`` place the hand
        in native MuJoCo coordinates (Z-up, quaternion order wxyz).

    The hand root is fixed to the frame at its origin, matching robot-models'
    root convention. The scene's options are unchanged. Packaged actuators,
    equalities and collision definitions are retained; none are synthesized.
    Requires the optional ``robot-models[mujoco]`` dependency.
    """
    cls = registry._model_class(name, "numpy")
    if not issubclass(cls, Hand):
        raise TypeError(f"{name!r} is not a hand")
    if side not in cls.supported_sides:
        raise ValueError(f"{name!r} supports side(s): {cls.supported_sides}")
    mujoco = cast(Any, import_module("mujoco"))  # Native bindings do not provide type stubs.
    directory = assets.model_dir(cls._NAME, model_path)
    hand = mujoco.MjSpec.from_file(str(directory / f"{side}.xml"))
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
    (root,) = hand.worldbody.bodies
    root.pos = [0, 0, 0]
    root.quat = [1, 0, 0, 0]
    for joint in list(root.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            hand.delete(joint)
    frame = spec.worldbody.add_frame() if frame is None else frame
    prefix = f"{cls._NAME}_{side}_" if prefix is None else prefix
    return spec.attach(hand, frame=frame, prefix=prefix)
