"""Static definitions shared by the catalog and model families."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import inf
from typing import Literal

Family = Literal["hands", "humanoids"]
Side = Literal["left", "right"]


@dataclass(frozen=True)
class Dof:
    """Meaning of one pose coordinate; effort is a peak joint torque in N·m."""

    finger: Literal["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]
    role: Literal["flexion", "abduction", "rotation"]
    closing: Literal[-1, 0, 1]
    effort_limit: float = inf


@dataclass(frozen=True)
class ModelSpec:
    """One canonical robot's family, anatomical names and asset requirements."""

    class_name: str
    family: Family
    joints: Mapping[str, str]
    sides: tuple[Side, ...] = ()
    asset_version: int = 1
    mirror: tuple[tuple[str, str], ...] = ()
    dofs: Mapping[str, Dof] = field(default_factory=dict)
