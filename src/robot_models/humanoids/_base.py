"""Shared behavior for canonical humanoids."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jaxtyping import Float

from robot_models import _catalog as catalog
from robot_models._base import RigidBodyModel
from robot_models._constants import Joint

Array = Any


class Humanoid(RigidBodyModel):
    """Floating-base humanoid posed by ``body_pose``."""

    _POSE = "body_pose"

    def __init__(self, *, model_path: Path | str | None = None) -> None:
        """Load the humanoid; ``model_path`` is an explicit asset directory."""
        self._setup(None, model_path, None, None)

    @property
    def common_joints(self) -> Mapping[Joint, str]:
        return {Joint(key): value for key, value in catalog.model_spec(self.name).joints.items()}

    @property
    def symmetric_bodies(self) -> tuple[tuple[int, int], ...]:
        """Left/right body pairs as ``(left_index, right_index)`` into :attr:`body_names`; unpaired bodies lie
        on the midline. ValueError when a sided body has no counterpart."""
        names = self.body_names
        pairs = []
        for index, name in enumerate(names):
            mirrored = name
            for left, right in catalog.model_spec(self.name).mirror:
                mirrored = mirrored.replace(left, right)
            if mirrored != name:
                if mirrored not in names:
                    raise ValueError(f"{type(self).__name__} body {name!r} has no counterpart {mirrored!r}")
                pairs.append((index, names.index(mirrored)))
        return tuple(pairs)

    def forward_skeleton(
        self,
        body_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch B 4 4"]:
        """World transforms of every body (:attr:`body_names`); the root at the origin by default."""
        return self._forward_skeleton(body_pose, global_rotation, global_translation)

    def to_qpos(
        self,
        body_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        scene_transform: Float[Array, "*batch 4 4"] | None = None,
    ) -> Float[Array, "*batch qpos"]:
        """MuJoCo ``qpos`` of the humanoid with a free root: ``[root_xyz, root_wxyz, hinge_angles]``.

        ``scene_transform`` is the rigid 4×4 transform ``(A, b)`` from the frame of ``global_rotation``
        (``R``) and ``global_translation`` (``t``) to the scene's world, the identity by default: the
        root goes to ``A t + b``, oriented ``A R M`` (``M`` turns MuJoCo's Z-up axes into the model's
        Y-up ones). The angles of :attr:`hinge_names` follow.
        """
        return self._to_qpos(body_pose, global_rotation, global_translation, scene_transform)
