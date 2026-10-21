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

# MANO's hinges, per joint: its bend, spread and twist hinges' peak torque (N·m) and enforced limits (degrees).
# Torques are nominal maximal voluntary torques: fingertip strength (40-50 N for the index) times each joint's lever,
# weaker for the ring and pinky; a finger's spread at its base matches its bend (Li et al. 2003, Clin Biomech 18:908).
# Hinges no muscle turns hold half their joint's bend, as its ligaments do. Limits are human maximal ranges widened
# generously, beyond any real hand: 99.9% of 237k synthesized MANO grasps (six methods) lie within, the others bend
# a joint where no hand can (a finger folded back flat).
MANO_HINGES: dict[str, tuple[tuple[float, float, float], ...]] = {
    "index1": ((3.0, -90, 150), (2.3, -60, 60), (1.5, -60, 60)),
    "index2": ((1.5, -60, 150), (0.75, -45, 45), (0.75, -60, 60)),
    "index3": ((0.7, -75, 135), (0.35, -45, 45), (0.35, -60, 60)),
    "middle1": ((3.0, -90, 150), (2.3, -60, 60), (1.5, -60, 60)),
    "middle2": ((1.5, -60, 150), (0.75, -45, 45), (0.75, -60, 60)),
    "middle3": ((0.7, -75, 135), (0.35, -45, 45), (0.35, -60, 60)),
    "pinky1": ((1.5, -90, 150), (1.2, -60, 60), (0.75, -60, 60)),
    "pinky2": ((0.8, -60, 150), (0.4, -45, 45), (0.4, -60, 60)),
    "pinky3": ((0.4, -75, 135), (0.2, -45, 45), (0.2, -60, 60)),
    "ring1": ((2.0, -90, 150), (1.5, -60, 60), (1.0, -60, 60)),
    "ring2": ((1.0, -60, 150), (0.5, -45, 45), (0.5, -60, 60)),
    "ring3": ((0.5, -75, 135), (0.25, -45, 45), (0.25, -60, 60)),
    "thumb1": ((3.0, -75, 150), (2.0, -60, 90), (1.0, -75, 120)),
    "thumb2": ((2.0, -90, 105), (1.0, -75, 60), (1.0, -90, 60)),
    "thumb3": ((1.0, -75, 135), (0.5, -60, 90), (0.5, -60, 90)),
}
MANO_ROLES = (("bend", "flexion"), ("spread", "abduction"), ("twist", "rotation"))


@dataclass(frozen=True)
class Source:
    """How one skinned robot is built from a Body Models model.

    ``hinges`` gives, for the mean-shape rest joints of a variant, each joint's three hinge
    axes in rotation order (rows) and their names and enforced ranges (none: unlimited);
    ``efforts`` the hinges' actuator torque bounds, in N·m.
    """

    body_model: str
    pose: str  # Body Models' pose parameter of the joints below the root
    root_rotation: str  # Body Models' rotation of the root joint about itself
    variant_key: str  # Body Models' argument choosing a variant: its model's first variant is the build's base
    density: float
    concavity: Callable[[str], float]
    hinges: Callable[[str, Float[np.ndarray, "J 3"], list[int], list[str]], list[tuple[np.ndarray, list[str], Any]]]
    options: Callable[[str], Mapping[str, Any]]
    mirrored: tuple[str, ...] = ()  # variants whose pieces mirror the base's
    free: bool = False
    actuated: bool = False
    efforts: Mapping[str, float] = field(default_factory=dict, compare=False)

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
        limits = np.radians([(low, high) for _, low, high in MANO_HINGES[name]])
        out.append((axes, [f"{name}_{role}" for role, _ in MANO_ROLES], limits))
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
        density=1100.0,  # soft tissue and bone
        concavity=lambda link: 0.04 if link in ("wrist", "thumb1") else 0.06,
        hinges=_mano_hinges,
        options=lambda side: {"side": side, "flat_hand_mean": True, "flip_shapedirs": side == "left"},
        mirrored=("left",),
        actuated=True,
        efforts={
            f"{joint}_{hinge}": effort
            for joint, hinges in MANO_HINGES.items()
            for (hinge, _), (effort, _, _) in zip(MANO_ROLES, hinges, strict=True)
        },
    ),
    "smpl": Source(
        body_model="smpl",
        pose="body_pose",
        root_rotation="pelvis_rotation",
        variant_key="gender",
        density=1000.0,
        concavity=lambda link: 0.05,
        hinges=_smpl_hinges,
        options=lambda gender: {"gender": gender},
        free=True,
    ),
}


@functools.cache
def numpy_model(source: Source, variant: str) -> Any:
    """The NumPy Body Models model of a source's variant, shared."""
    return source.create(variant)
