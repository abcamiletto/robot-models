"""Materialize immutable model data for an array backend.

Dataclasses recurse (a module for PyTorch, a pytree for JAX), NumPy arrays become the
backend's arrays, and every other value (names, indices) stays as it is, static.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np

_JAX_DATACLASSES: set[type] = set()


def numpy_state(value: Any) -> Any:
    """Materialize model data for NumPy: copies of its arrays."""
    if is_dataclass(value):
        return type(value)(**{field.name: numpy_state(getattr(value, field.name)) for field in fields(value)})
    return value.copy() if isinstance(value, np.ndarray) else value


def torch_state(value: Any) -> Any:
    """Materialize model data for PyTorch: modules whose arrays are buffers."""
    import torch
    from torch import nn

    if is_dataclass(value):
        module = nn.Module()
        for field in fields(value):
            setattr(module, field.name, torch_state(getattr(value, field.name)))
        return module
    return nn.Buffer(torch.tensor(value), persistent=True) if isinstance(value, np.ndarray) else value


def jax_state(value: Any) -> Any:
    """Materialize model data for JAX: pytrees of JAX arrays."""
    import jax.numpy as jnp

    if is_dataclass(value):
        register_jax_state(value)
        return type(value)(**{field.name: jax_state(getattr(value, field.name)) for field in fields(value)})
    return jnp.asarray(value) if isinstance(value, np.ndarray) else value


def register_jax_state(value: Any) -> None:
    """Register a dataclass, and those it holds, as JAX pytrees whose non-array leaves are static."""
    if not is_dataclass(value):
        return
    for field in fields(value):
        register_jax_state(getattr(value, field.name))
    cls = type(value)
    if cls in _JAX_DATACLASSES:
        return
    import jax

    names = tuple(field.name for field in fields(cls))

    def flatten(obj: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        leaves, tree = jax.tree_util.tree_flatten(tuple(getattr(obj, name) for name in names))
        static = tuple((index, leaf) for index, leaf in enumerate(leaves) if not isinstance(leaf, jax.Array))
        return tuple(leaf for leaf in leaves if isinstance(leaf, jax.Array)), (tree, static)

    def unflatten(aux: tuple[Any, ...], children: tuple[Any, ...]) -> Any:
        tree, static = aux
        leaves = list(children)
        for index, leaf in static:
            leaves.insert(index, leaf)
        return cls(**dict(zip(names, jax.tree_util.tree_unflatten(tree, leaves), strict=True)))

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    _JAX_DATACLASSES.add(cls)


__all__ = ["jax_state", "numpy_state", "register_jax_state", "torch_state"]
