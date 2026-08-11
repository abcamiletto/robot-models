"""Array runtimes for backend-independent model programs.

Runtime methods lower backend-independent operations at call time. Reusable
derived inputs belong to backend-materialized state instead; see
:mod:`robot_models._state`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeAlias

import numpy as np
from jaxtyping import Float, Num

from robot_models import _common as common
from robot_models import _state as state

Array = Any
RuntimeName: TypeAlias = Literal["numpy", "torch", "jax"]


class ArrayRuntime(ABC):
    """Shared numerical operations for one array backend."""

    name: ClassVar[RuntimeName]

    @property
    @abstractmethod
    def xp(self) -> Any:
        """Array namespace for this runtime."""

    def asarray(
        self,
        value: Any,
        *,
        like: Num[Array, "..."],
        dtype: Any | None = None,
    ) -> Num[Array, "..."]:
        """Create an array with the backend, device, and default dtype of ``like``."""
        if dtype is None:
            dtype = like.dtype
        return self.xp.asarray(value, dtype=dtype)

    def zeros(
        self,
        shape: tuple[int, ...],
        *,
        like: Float[Array, "..."],
        dtype: Any | None = None,
    ) -> Float[Array, "..."]:
        """Create zeros with the backend and device of ``like``."""
        return common.zeros_as(like, shape=shape, dtype=dtype, xp=self.xp)

    @abstractmethod
    def _materialize(self, value: Any) -> Any:
        """Convert loaded model data into backend-managed state."""

    @abstractmethod
    def stop_gradient(self, value: Num[Array, "..."]) -> Num[Array, "..."]:
        """Return ``value`` without gradient propagation."""

    @abstractmethod
    def to_numpy(self, value: Num[Array, "..."]) -> Num[np.ndarray, "..."]:
        """Convert an array to NumPy host memory."""


@dataclass(frozen=True)
class NumpyRuntime(ArrayRuntime):
    """NumPy model runtime."""

    name = "numpy"

    @property
    def xp(self) -> Any:
        return np

    def _materialize(self, value: Any) -> Any:
        return state.numpy_state(value)

    def stop_gradient(self, value: Num[Array, "..."]) -> Num[Array, "..."]:
        return value

    def to_numpy(self, value: Num[Array, "..."]) -> Num[np.ndarray, "..."]:
        return np.asarray(value)


@dataclass(frozen=True)
class TorchRuntime(ArrayRuntime):
    """Torch array runtime."""

    name = "torch"

    @property
    def xp(self) -> Any:
        import torch

        return torch

    def asarray(
        self,
        value: Any,
        *,
        like: Num[Array, "..."],
        dtype: Any | None = None,
    ) -> Num[Array, "..."]:
        if dtype is None:
            dtype = like.dtype
        return self.xp.as_tensor(value, device=like.device, dtype=dtype)

    def _materialize(self, value: Any) -> Any:
        return state.torch_state(value)

    def stop_gradient(self, value: Num[Array, "..."]) -> Num[Array, "..."]:
        return value.detach()

    def to_numpy(self, value: Num[Array, "..."]) -> Num[np.ndarray, "..."]:
        return value.detach().cpu().numpy()


@dataclass(frozen=True)
class JaxRuntime(ArrayRuntime):
    """JAX model runtime."""

    name = "jax"

    @property
    def xp(self) -> Any:
        import jax.numpy as jnp

        return jnp

    def asarray(
        self,
        value: Any,
        *,
        like: Num[Array, "..."],
        dtype: Any | None = None,
    ) -> Num[Array, "..."]:
        import jax

        if dtype is None:
            dtype = like.dtype
        array = self.xp.asarray(value, dtype=dtype)
        device = getattr(like, "device", None)
        return array if device is None else jax.device_put(array, device)

    def _materialize(self, value: Any) -> Any:
        return state.jax_state(value)

    def stop_gradient(self, value: Num[Array, "..."]) -> Num[Array, "..."]:
        import jax

        return jax.lax.stop_gradient(value)

    def to_numpy(self, value: Num[Array, "..."]) -> Num[np.ndarray, "..."]:
        import jax

        return np.asarray(jax.device_get(value))


__all__ = [
    "ArrayRuntime",
    "JaxRuntime",
    "NumpyRuntime",
    "RuntimeName",
    "TorchRuntime",
]
