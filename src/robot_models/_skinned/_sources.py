"""Body Models sources of the skinned robots: model files, hinge axes and ranges, tissue density."""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float

# Reflection of a left hand into the right one (and back): MANO's left model mirrors the right across x.
MIRROR = np.diag([-1.0, 1.0, 1.0])

# MANO's joint limits, degrees, (low, high) of bend, spread and twist per joint of a finger and of the thumb.
# Enforced in MuJoCo: human maximal ranges widened generously, beyond any real hand; 99.9% of 237k synthesized
# MANO grasps (six methods) lie within, the others bend a joint where no hand can (a finger folded back flat).
_LIMITS = {
    "finger": (
        ((-90, 150), (-60, 60), (-60, 60)),
        ((-60, 150), (-45, 45), (-60, 60)),
        ((-75, 135), (-45, 45), (-60, 60)),
    ),
    "thumb": (
        ((-75, 150), (-60, 90), (-75, 120)),
        ((-90, 105), (-75, 60), (-90, 60)),
        ((-75, 135), (-60, 90), (-60, 90)),
    ),
}
# Anatomical, after manotorch (AnatomyConstraintLossEE.setup): how far a pose lies from a natural hand's.
_ANATOMY = {
    "finger": (((0, 90), (-5, 5), (0, 0)), ((0, 90), (0, 0), (0, 0)), ((0, 90), (0, 0), (0, 0))),
    "thumb": (((0, 45), (-15, 45), (-45, 45)), ((0, 90), (-10, 10), (0, 0)), ((0, 90), (0, 0), (0, 0))),
}
ROLES = ("bend", "spread", "twist")
MANO_ANATOMY = {
    f"{finger}{joint}_{role}": np.radians(_ANATOMY["thumb" if finger == "thumb" else "finger"][joint - 1][axis])
    for finger in ("index", "middle", "pinky", "ring", "thumb")
    for joint in (1, 2, 3)
    for axis, role in enumerate(ROLES)
}


@dataclass(frozen=True)
class Source:
    """How one skinned robot is built from a Body Models model.

    ``hinges`` gives, for the mean-shape rest joints of a variant, each joint's three hinge
    axes in rotation order (rows) and their names and enforced ranges (none: unlimited).
    """

    body_model: str
    pose: str  # Body Models' pose parameter of the joints below the root
    root_rotation: str  # Body Models' rotation of the root joint about itself
    variant_key: str  # Body Models' argument choosing a variant
    variants: tuple[str, ...]
    base: str
    density: float
    concavity: Callable[[str], float]
    hinges: Callable[[str, Float[np.ndarray, "J 3"], list[int], list[str]], list[tuple[np.ndarray, list[str], Any]]]
    options: Callable[[str], Mapping[str, Any]]
    mirrored: tuple[str, ...] = ()  # variants whose pieces mirror the base's
    anatomy: Mapping[str, np.ndarray] = field(default_factory=dict)  # natural ranges by hinge name, radians
    free: bool = False
    actuated: bool = False

    def create(self, variant: str, runtime: Literal["numpy", "torch", "jax"] = "numpy") -> Any:
        """The Body Models model of a variant, from its configured local file."""
        from body_models import create_model

        return create_model(self.body_model, runtime=runtime, **self.options(variant))

    def path(self, variant: str) -> Path:
        """The configured model file of a variant; FileNotFoundError when Body Models has none."""
        io = import_module(f"body_models.{self.body_model}._io")
        return Path(io.get_model_path(None, **{self.variant_key: variant})).resolve()


def _mano_hinges(variant: str, joints: np.ndarray, parents: list[int], names: list[str]) -> list:
    """Bend, spread and twist axes from manotorch's anatomical frames of the right template.

    Twist runs along the bone towards its parent, bend is twist x up (up is y for the fingers,
    (1, 1, 1) for the thumb) and spread completes the frame. A left hand's axes are the mirrored
    right ones, negated as a reflection maps a rotation axis, so its angles mean the same motion.
    """
    sign = MIRROR if variant == "left" else np.eye(3)
    right = joints @ sign
    out = []
    for joint, name in enumerate(names):
        parent = parents[joint]
        if parent < 0:
            continue
        twist = right[parent] - right[joint]
        up = np.array([1.0, 1.0, 1.0]) if name.startswith("thumb") else np.array([0.0, 1.0, 0.0])
        bend = np.cross(twist, up)
        spread = np.cross(bend, twist)
        axes = np.stack([bend, spread, twist]) / np.linalg.norm([bend, spread, twist], axis=1, keepdims=True)
        if variant == "left":
            axes = -axes @ MIRROR
        limits = _LIMITS["thumb" if name.startswith("thumb") else "finger"][int(name[-1]) - 1]
        out.append((axes, [f"{name}_{role}" for role in ROLES], np.radians(limits)))
    return out


def _smpl_hinges(variant: str, joints: np.ndarray, parents: list[int], names: list[str]) -> list:
    """Rotations about the body frame's x, y and z axes, in that order, without limits."""
    return [(np.eye(3), [f"{name}_{axis}" for axis in "xyz"], None) for name, p in zip(names, parents) if p >= 0]


SOURCES: dict[str, Source] = {
    "mano": Source(
        body_model="mano",
        pose="hand_pose",
        root_rotation="wrist_rotation",
        variant_key="side",
        variants=("right", "left"),
        base="right",
        density=1100.0,  # soft tissue and bone
        concavity=lambda link: 0.04 if link in ("wrist", "thumb1") else 0.06,
        hinges=_mano_hinges,
        options=lambda side: {"side": side, "flat_hand_mean": True, "flip_shapedirs": side == "left"},
        mirrored=("left",),
        anatomy=MANO_ANATOMY,
        actuated=True,
    ),
    "smpl": Source(
        body_model="smpl",
        pose="body_pose",
        root_rotation="pelvis_rotation",
        variant_key="gender",
        variants=("neutral", "male", "female"),
        base="neutral",
        density=1000.0,
        concavity=lambda link: 0.05,
        hinges=_smpl_hinges,
        options=lambda gender: {"gender": gender},
        free=True,
    ),
}


def source_of(name: str) -> Source:
    """The Body Models source of a locally built model."""
    from robot_models import _catalog as catalog

    body_model = catalog.model_spec(name).body_model
    if body_model is None:
        raise ValueError(f"{name!r} is not built from Body Models")
    return SOURCES[body_model]


@functools.cache
def numpy_model(name: str, variant: str) -> Any:
    """The NumPy Body Models model of a locally built model's variant, shared."""
    return source_of(name).create(variant)
