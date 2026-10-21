"""Shared behavior for canonical hands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from jaxtyping import Float

from robot_models import _catalog as catalog
from robot_models._base import RigidBodyModel
from robot_models._constants import Joint
from robot_models._spec import Dof, Side, sided

Array = Any


class Hand(RigidBodyModel):
    """Single robot hand posed by ``hand_pose``."""

    _POSE = "hand_pose"

    def __init__(
        self,
        *,
        side: Side = "right",
        kinematics: str | None = None,
        shape: Sequence[float] | None = None,
        conform_to: Sequence[float] | None = None,
        model_path: Path | str | None = None,
    ) -> None:
        """Load one side of the hand.

        ``kinematics`` selects another description's joint placements and ranges on the same
        bodies, joints and meshes, such as Shadow's ``"openai"``; ``None`` keeps the hand's own.
        ``shape`` and ``conform_to`` fix the geometry of a hand built from Body Models (MANO): its
        shape coefficients (the mean by default) and the ``hand_pose`` its links are cut at (flat
        by default). ``model_path`` is an explicit asset directory.
        """
        self._setup(side, model_path, shape, conform_to, kinematics)

    @property
    def side(self) -> Side:
        return cast(Side, self._config.variant)

    @property
    def kinematics(self) -> str | None:
        """The other kinematics the hand was loaded with, ``None`` for its own."""
        return self._config.kinematics

    @property
    def dofs(self) -> tuple[Dof, ...]:
        """Coordinate semantics, aligned with :attr:`dof_names`."""
        definitions = catalog.model_spec(self.name).dofs
        by_name = {sided(name, self.side): dof for name, dof in definitions.items()}
        return tuple(by_name[name] for name in self.dof_names)

    @property
    def effort_limits(self) -> Float[Array, "Q"]:
        """Each coordinate's peak torque in N·m, its MJCF ``actuatorfrcrange``; through a coupling, it drives the
        coupled hinges too."""
        return self._assets.dof_efforts

    @property
    def closing(self) -> Float[Array, "Q"]:
        """Each coordinate's closing direction: 1 for a finger's flexion, 0 for every other coordinate."""
        flexing = [dof.role == "flexion" and dof.finger != "wrist" for dof in self.dofs]
        return self.runtime.asarray(flexing, like=self.dof_limits)

    def close(self, hand_pose: Float[Array, "*batch Q"], angle: float | Array) -> Float[Array, "*batch Q"]:
        """Add radians along :attr:`closing`, within :attr:`dof_limits`. Negative angles open.

        ``angle`` is scalar or has batch dimensions. Other coordinates stay unchanged.
        """
        self._check(hand_pose)
        xp = self.runtime.xp
        direction = self.closing
        amount = self.runtime.asarray(angle, like=hand_pose)
        lower, upper = self.dof_limits.T
        closed = xp.clip(hand_pose + amount[..., None] * direction, lower, upper)
        return xp.where(direction != 0, closed, hand_pose)

    @property
    def common_joints(self) -> Mapping[Joint, str]:
        joints = catalog.model_spec(self.name).joints
        return {Joint(f"{self.side}_{key}"): sided(name, self.side) for key, name in joints.items()}

    def forward_skeleton(
        self,
        hand_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch B 4 4"]:
        """World transforms of every body (:attr:`body_names`); the root at the origin by default."""
        return self._forward_skeleton(hand_pose, global_rotation, global_translation)

    def to_qpos(
        self,
        hand_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        scene_transform: Float[Array, "*batch 4 4"] | None = None,
    ) -> Float[Array, "*batch qpos"]:
        """MuJoCo ``qpos`` of the hand with a free root: ``[root_xyz, root_wxyz, hinge_angles]``.

        ``scene_transform`` is the rigid 4×4 transform ``(A, b)`` from the frame of ``global_rotation``
        (``R``) and ``global_translation`` (``t``) to the scene's world, the identity by default: the
        root goes to ``A t + b``, oriented ``A R M`` (``M`` turns MuJoCo's Z-up axes into the model's
        Y-up ones). The angles of :attr:`hinge_names` follow.
        """
        return self._to_qpos(hand_pose, global_rotation, global_translation, scene_transform)

    def _preset_key(self, preset: str) -> str:
        return f"{self.side}_{preset}"
