import model_cases
import numpy as np
import pytest


def link_loss(model, params):
    positions = model.forward_links(**params)[..., :3, 3]
    return (positions**2).sum()


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_torch_and_jax_gradients_match_finite_difference(name, model_class, kwargs) -> None:
    torch = pytest.importorskip("torch")
    torch_model = model_cases.backend_model_class(name, "torch")(**kwargs).double()
    torch_rest = {key: value + 0.03 for key, value in torch_model.get_rest_pose(dtype=torch.float64).items()}

    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    jax_model = model_cases.backend_model_class(name, "jax")(**kwargs)
    jax_rest = {key: value + 0.03 for key, value in jax_model.get_rest_pose(dtype=jnp.float64).items()}

    for key in torch_rest:
        torch_params = {parameter: value.detach() for parameter, value in torch_rest.items()}
        torch_value = torch_params[key].clone().requires_grad_(True)
        torch_params[key] = torch_value
        link_loss(torch_model, torch_params).backward()
        torch_auto = torch_value.grad.reshape(-1)[0].item()

        def jax_loss(value, parameter=key):
            params = jax_rest | {parameter: value}
            return link_loss(jax_model, params)

        jax_value = jax_rest[key]
        jax_auto = np.asarray(jax.grad(jax_loss)(jax_value)).reshape(-1)[0]
        np.testing.assert_allclose(jax_auto, torch_auto, rtol=1e-3, atol=1e-3, err_msg=f"{name}.{key}")
