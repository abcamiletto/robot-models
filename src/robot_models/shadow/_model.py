"""Shadow Dexterous Hand."""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from robot_models import _assets as assets
from robot_models._base import Hand, RigidBodyModel, Side
from robot_models._catalog import MODEL_SOURCES
from robot_models._constants import Joint
from robot_models._runtime import ArrayRuntime
from robot_models.shadow import _source


class ShadowHand(Hand):
    """Shadow Hand with an explicit description source and native pose conventions."""

    _NAME = "shadow"
    available_sources = tuple(MODEL_SOURCES[_NAME])
    # Two-DoF anatomical joints map to the distal body of each coincident pair.
    _HAND_JOINTS = MappingProxyType(
        {
            "wrist": "{s}h_palm",
            "thumb_cmc": "{s}h_thproximal",
            "thumb_mcp": "{s}h_thmiddle",
            "thumb_ip": "{s}h_thdistal",
            "index_mcp": "{s}h_ffproximal",
            "index_pip": "{s}h_ffmiddle",
            "index_dip": "{s}h_ffdistal",
            "middle_mcp": "{s}h_mfproximal",
            "middle_pip": "{s}h_mfmiddle",
            "middle_dip": "{s}h_mfdistal",
            "ring_mcp": "{s}h_rfproximal",
            "ring_pip": "{s}h_rfmiddle",
            "ring_dip": "{s}h_rfdistal",
            "pinky_mcp": "{s}h_lfproximal",
            "pinky_pip": "{s}h_lfmiddle",
            "pinky_dip": "{s}h_lfdistal",
        }
    )

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        side: Side = "right",
        source: Literal["menagerie", "gendexgrasp", "dexgraspnet"] = "menagerie",
        runtime: ArrayRuntime,
    ) -> None:
        if source not in self.available_sources:
            raise ValueError(f"Unknown source {source!r}. Available sources: {self.available_sources}")
        supported = self.supported_sides if source == "menagerie" else ("right",)
        if side not in supported:
            raise ValueError(f"Shadow source {source!r} supports side(s): {supported}")
        directory = assets.model_dir(self._NAME, model_path, source=source)
        _source.validate_source(directory, source)
        super().__init__(model_path=directory, side=side, runtime=runtime)
        # Source and side are static model state, preserved by pickle and JAX pytrees.
        self._config = (side, source)

    @property
    def side(self) -> Side:
        return self._config if isinstance(self._config, str) else self._config[0]

    @property
    def source(self) -> str:
        # Older canonical pickles/JAX auxiliary state stored just the side.
        return "menagerie" if isinstance(self._config, str) else self._config[1]

    @property
    def common_joints(self) -> Mapping[Joint, str]:
        if self.source == "menagerie":
            return super().common_joints
        names = {key: name.removeprefix("{s}h_") for key, name in self._HAND_JOINTS.items()}
        if self.source == "dexgraspnet":
            names = {key: f"robot0:{name}" + ("" if key == "wrist" else "_child") for key, name in names.items()}
        return {Joint(f"{self.side}_{key}"): name for key, name in names.items()}

    def get_rest_pose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Any]:
        if self.source == "menagerie":
            return super().get_rest_pose(batch_dims=batch_dims, dtype=dtype, hands=hands)
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Unknown hand preset: {hands!r}")
        params = RigidBodyModel.get_rest_pose(self, batch_dims=batch_dims, dtype=dtype)
        if hands != "default":
            lo, hi = self.actuated_joint_limits[:, 0], self.actuated_joint_limits[:, 1]
            params["hand_pose"] = self.runtime.xp.clip(params["hand_pose"], lo, hi)
        return params


__all__ = ["ShadowHand"]
