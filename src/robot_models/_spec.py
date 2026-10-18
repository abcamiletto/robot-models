"""Static definitions shared by the catalog and model families."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import inf
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from robot_models._skinned._sources import Source

Family = Literal["hands", "humanoids"]
Side = Literal["left", "right"]
Finger = Literal["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]
Role = Literal["flexion", "abduction", "rotation"]


@dataclass(frozen=True)
class Dof:
    """Meaning of one pose coordinate; effort is a peak joint torque in N·m."""

    finger: Finger
    role: Role
    effort_limit: float = inf


@dataclass(frozen=True)
class ModelSpec:
    """One canonical robot's family, anatomical names and asset requirements.

    ``variants`` are the MJCF files a model offers, ``{variant}.xml``: a hand's sides, SMPL's
    genders; none for a humanoid of one file, ``{name}.xml``. Catalog names may hold the side as
    ``{side}`` or its initial as ``{s}``. ``source`` is the Body Models model a model's assets are
    built from locally, instead of downloaded; its assets are never redistributed.
    """

    class_name: str
    family: Family
    joints: Mapping[str, str]
    variants: tuple[str, ...] = ()
    asset_version: int = 1
    mirror: tuple[tuple[str, str], ...] = ()
    dofs: Mapping[str, Dof] = field(default_factory=dict)
    source: Source | None = None


def sided(name: str, side: str) -> str:
    """A catalog name of one side."""
    return name.format(side=side, s=side[0])
