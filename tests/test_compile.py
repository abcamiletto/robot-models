import model_cases
import numpy as np
import pytest


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.RUNTIME_MODELS)
def test_torch_compile_and_jax_jit(name, model_class, kwargs) -> None:
    torch = pytest.importorskip("torch")
    torch._dynamo.reset()
    torch_class = model_cases.backend_model_class(name, "torch")
    torch_model = torch_class(**kwargs)
    torch_params = torch_model.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    torch_params = {key: value + 0.03 for key, value in torch_params.items()}
    with torch.no_grad():
        expected = torch_model.forward_links(**torch_params)
        torch_links = torch.compile(torch_model.forward_links, backend="eager", fullgraph=True)(**torch_params)
    torch.testing.assert_close(torch_links, expected)

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax_class = model_cases.backend_model_class(name, "jax")
    jax_model = jax_class(**kwargs)
    jax_params = jax_model.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    jax_params = {key: value + 0.03 for key, value in jax_params.items()}
    jax_links = jax.jit(jax_model.forward_links)(**jax_params)
    np.testing.assert_allclose(jax_links, jax_model.forward_links(**jax_params), atol=1e-6, rtol=1e-6)
