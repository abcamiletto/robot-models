"""Model runtime behavior."""

import pickle

import model_cases
import numpy as np
import pytest

from robot_models import create_model


def test_torch_model_manages_module_state() -> None:
    torch = pytest.importorskip("torch")
    model = create_model("g1", runtime="torch").double()
    for geometry in ("visual", "collision"):
        meshes = getattr(model, geometry).meshes
        original = meshes[0].vertices[0].copy()
        meshes[0].vertices[0] += 1
        np.testing.assert_array_equal(getattr(model, geometry).meshes[0].vertices[0], original)

    params = model.get_rest_pose(dtype=torch.float64)
    expected = model.forward_skeleton(**params)
    assert expected.dtype == torch.float64
    assert model.state_dict()
    assert all(buffer.dtype == torch.float64 for buffer in model.buffers() if buffer.is_floating_point())
    restored = pickle.loads(pickle.dumps(model))
    assert type(restored) is type(model)
    assert restored.name == "g1"
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.forward_skeleton(**params), expected)


@pytest.mark.parametrize(("name", "kwargs"), model_cases.RUNTIME_CASES)
def test_jax_model_pytree_round_trip(name, kwargs) -> None:
    jax = pytest.importorskip("jax")
    model = create_model(name, runtime="jax", **kwargs)

    leaves, tree = jax.tree_util.tree_flatten(model)
    restored = jax.tree_util.tree_unflatten(tree, leaves)

    assert type(restored) is type(model)
    assert restored.runtime == model.runtime
    assert restored.name == name
    assert restored.body_names == model.body_names
    if hasattr(model, "side"):
        assert restored.side == model.side
    params = model.get_rest_pose()
    np.testing.assert_allclose(restored.forward_skeleton(**params), model.forward_skeleton(**params))
