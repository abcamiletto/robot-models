import model_cases
import numpy as np
import pytest

from robot_models import create_model


@pytest.mark.parametrize(("name", "kwargs"), model_cases.RUNTIME_CASES)
def test_torch_compile_and_jax_jit(name, kwargs) -> None:
    torch = pytest.importorskip("torch")
    torch._dynamo.reset()
    torch_model = create_model(name, runtime="torch", **kwargs)
    torch_params = torch_model.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    torch_params = {key: value + 0.03 for key, value in torch_params.items()}
    with torch.no_grad():
        expected = torch_model.forward_skeleton(**torch_params)
        compiled = torch.compile(torch_model.forward_skeleton, backend="eager", fullgraph=True)(**torch_params)
    torch.testing.assert_close(compiled, expected)

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax_model = create_model(name, runtime="jax", **kwargs)
    jax_params = jax_model.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    jax_params = {key: value + 0.03 for key, value in jax_params.items()}
    jitted = jax.jit(jax_model.forward_skeleton)(**jax_params)
    np.testing.assert_allclose(jitted, jax_model.forward_skeleton(**jax_params), atol=1e-6, rtol=1e-6)
