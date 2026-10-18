import model_cases
import numpy as np
import pytest

from robot_models import create_model


def link_loss(model, params):
    positions = model.forward_skeleton(**params)[..., :3, 3]
    return (positions**2).sum()


@pytest.mark.parametrize(("name", "kwargs"), model_cases.RUNTIME_CASES)
def test_torch_and_jax_gradients_match_finite_difference(name, kwargs) -> None:
    torch = pytest.importorskip("torch")
    torch_model = create_model(name, runtime="torch", **kwargs).double()
    torch_rest = {key: value + 0.03 for key, value in torch_model.get_rest_pose(dtype=torch.float64).items()}

    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    jax_model = create_model(name, runtime="jax", **kwargs)
    jax_rest = {key: value + 0.03 for key, value in jax_model.get_rest_pose(dtype=jnp.float64).items()}

    for key in torch_rest:
        torch_params = {parameter: value.detach() for parameter, value in torch_rest.items()}
        torch_value = torch_params[key].clone().requires_grad_(True)
        torch_params[key] = torch_value
        link_loss(torch_model, torch_params).backward()
        torch_auto = torch_value.grad.detach().numpy()

        def jax_loss(value, parameter=key):
            params = jax_rest | {parameter: value}
            return link_loss(jax_model, params)

        jax_value = jax_rest[key]
        jax_auto = np.asarray(jax.grad(jax_loss)(jax_value))
        np.testing.assert_allclose(jax_auto, torch_auto, rtol=1e-3, atol=1e-3, err_msg=f"{name}.{key}")
        delta = np.zeros_like(jax_auto)
        delta.flat[0] = 1e-5
        finite_difference = (jax_loss(jax_value + delta) - jax_loss(jax_value - delta)) / 2e-5
        np.testing.assert_allclose(jax_auto.flat[0], finite_difference, rtol=1e-3, atol=1e-3, err_msg=f"{name}.{key}")
