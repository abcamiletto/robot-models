"""Static definitions shared by the catalog and model families."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Family = Literal["hands", "humanoids"]
Side = Literal["left", "right"]


@dataclass(frozen=True)
class ModelSpec:
    """One canonical robot's family, anatomical names and asset requirements."""

    class_name: str
    family: Family
    joints: Mapping[str, str]
    sides: tuple[Side, ...] = ()
    asset_version: int = 1
    mirror: tuple[tuple[str, str], ...] = ()
