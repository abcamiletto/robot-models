"""Model runtime behavior."""

import pickle

import model_cases
import numpy as np
import pytest

from robot_models._runtime import JaxRuntime, NumpyRuntime, TorchRuntime


@pytest.mark.fast
def test_runtime_array_creation_follows_reference_dtype() -> None:
    reference = np.zeros((), dtype=np.float64)
    assert NumpyRuntime().asarray([1.0], like=reference).dtype == np.float64
    assert NumpyRuntime().zeros((2, 3), like=reference).dtype == np.float64

    torch = pytest.importorskip("torch")
    reference = torch.zeros((), dtype=torch.float64)
    assert TorchRuntime().asarray([1.0], like=reference).dtype == torch.float64
    assert TorchRuntime().zeros((2, 3), like=reference).dtype == torch.float64


@pytest.mark.fast
@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_runtime_rejects_unknown_state(backend) -> None:
    if backend != "numpy":
        pytest.importorskip(backend)
    runtime = {"numpy": NumpyRuntime, "torch": TorchRuntime, "jax": JaxRuntime}[backend]()
    with pytest.raises(TypeError, match="Unsupported model state leaf"):
        runtime._materialize(object())


@pytest.mark.fast
def test_runtime_stop_gradient() -> None:
    numpy_value = np.ones(2, dtype=np.float32)
    assert NumpyRuntime().stop_gradient(numpy_value) is numpy_value

    torch = pytest.importorskip("torch")
    assert not TorchRuntime().stop_gradient(torch.ones(2, requires_grad=True)).requires_grad

    jax = pytest.importorskip("jax")
    gradient = jax.grad(lambda value: JaxRuntime().stop_gradient(value).sum())(jax.numpy.ones(2))
    np.testing.assert_array_equal(gradient, np.zeros(2, dtype=np.float32))


def test_factory_returns_backend_specific_model() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("jax")
    from robot_models import create_model
    from robot_models.g1.jax import G1 as JaxG1
    from robot_models.g1.numpy import G1 as NumpyG1
    from robot_models.g1.torch import G1 as TorchG1

    models = [create_model("g1"), create_model("g1", runtime="torch"), create_model("g1", runtime="jax")]
    assert [type(model) for model in models] == [NumpyG1, TorchG1, JaxG1]
    assert [model.runtime.name for model in models] == ["numpy", "torch", "jax"]
    assert isinstance(models[1], torch.nn.Module)


def test_torch_model_manages_module_state() -> None:
    torch = pytest.importorskip("torch")
    from robot_models.g1.torch import G1

    model = G1().double()
    assert "_weights.vertices" in model.state_dict()
    assert model._weights.vertices.dtype == torch.float64

    restored = pickle.loads(pickle.dumps(model))
    assert isinstance(restored, G1)
    assert "_weights.vertices" in restored.state_dict()


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_jax_model_pytree_round_trip(name, model_class, kwargs) -> None:
    jax = pytest.importorskip("jax")
    model = model_cases.backend_model_class(name, "jax")(**kwargs)

    leaves, tree = jax.tree_util.tree_flatten(model)
    restored = jax.tree_util.tree_unflatten(tree, leaves)

    assert type(restored) is type(model)
    assert restored.runtime == model.runtime
    assert restored.joint_names == model.joint_names
