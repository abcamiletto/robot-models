import model_cases
import numpy as np
import pytest


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_torch_compile_and_jax_jit(name, model_class, kwargs) -> None:
    torch = pytest.importorskip("torch")
    torch_class = model_cases.backend_model_class(name, "torch")
    torch_model = torch_class(**kwargs)
    torch_params = torch_model.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    with torch.no_grad():
        torch_links = torch.compile(torch_model.forward_links, backend="eager", fullgraph=True)(**torch_params)
    assert torch_links.shape[-2:] == (4, 4)

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax_class = model_cases.backend_model_class(name, "jax")
    jax_model = jax_class(**kwargs)
    jax_params = jax_model.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    jax_links = jax.jit(jax_model.forward_links)(**jax_params)
    assert np.asarray(jax_links).shape[-2:] == (4, 4)
