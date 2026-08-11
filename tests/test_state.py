"""Backend model-state materialization behavior."""

from dataclasses import dataclass

import numpy as np
import pytest

from robot_models._state import jax_state, numpy_state, torch_state


@dataclass(frozen=True)
class _Leaf:
    values: np.ndarray


@dataclass(frozen=True)
class _Tree:
    leaves: dict[str, _Leaf]
    arrays: dict[str, np.ndarray]


@pytest.mark.fast
def test_torch_state_registers_nested_arrays() -> None:
    torch = pytest.importorskip("torch")
    state = torch_state(
        _Tree(
            leaves={"low": _Leaf(np.ones(2, dtype=np.float32))},
            arrays={"indices": np.arange(2)},
        )
    )

    assert list(state.state_dict()) == ["leaves.low.values", "arrays.indices"]
    state.to(dtype=torch.float64)
    assert state.leaves["low"].values.dtype == torch.float64


@pytest.mark.fast
def test_materialized_arrays_have_independent_storage() -> None:
    source = np.ones(2, dtype=np.float32)
    numpy = numpy_state(source)
    numpy[0] = 2
    assert source[0] == 1

    pytest.importorskip("torch")
    torch_array = torch_state(source)
    torch_array[0] = 3
    assert source[0] == 1


@pytest.mark.fast
def test_jax_state_preserves_jax_arrays() -> None:
    jax = pytest.importorskip("jax")
    value = jax.numpy.ones(2)
    assert jax_state(value) is value
