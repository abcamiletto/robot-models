"""Every runtime produces the same frames and meshes, for any batch shape."""

import model_cases
import numpy as np
import pytest
from trimesh import Trimesh

from robot_models import create_model, list_models

LEADING_DIM_BATCH_SHAPES = [(), (2,), (2, 2)]


def mesh_vertices(meshes):
    return np.stack([np.asarray(mesh.vertices) for mesh in meshes], axis=0)


@pytest.mark.parametrize("name", list_models())
def test_meshes_match_across_backends(name) -> None:
    numpy_model = create_model(name)
    numpy_params = numpy_model.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    key = model_cases.pose_name(numpy_model)
    lo, hi = np.asarray(numpy_model.dof_limits).T
    numpy_params[key] = np.clip(numpy_params[key] + np.array([[0.07], [0.13]], dtype=np.float32), lo, hi)

    torch = pytest.importorskip("torch")
    torch_model = create_model(name, runtime="torch")
    torch_params = {key: torch.as_tensor(value) for key, value in numpy_params.items()}

    pytest.importorskip("jax")
    import jax.numpy as jnp

    jax_model = create_model(name, runtime="jax")
    jax_params = {key: jnp.asarray(value) for key, value in numpy_params.items()}
    for geometry in ("visual", "collision"):
        expected = mesh_vertices(getattr(numpy_model, geometry).pose(numpy_model.forward_skeleton(**numpy_params)))
        with torch.no_grad():
            torch_meshes = getattr(torch_model, geometry).pose(torch_model.forward_skeleton(**torch_params))
        jax_meshes = getattr(jax_model, geometry).pose(jax_model.forward_skeleton(**jax_params))
        assert all(isinstance(mesh, Trimesh) for mesh in (*torch_meshes, *jax_meshes))
        np.testing.assert_allclose(mesh_vertices(torch_meshes), expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(mesh_vertices(jax_meshes), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name", list_models())
def test_rest_pose_has_every_parameter(name) -> None:
    model = create_model(name)
    params = model.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    assert {key: value.shape for key, value in params.items()} == {
        key: (2, *dims) for key, dims in model.parameter_spec.items()
    }
    assert all(not value.any() for value in params.values())


@pytest.mark.parametrize(("name", "kwargs"), model_cases.RUNTIME_CASES)
def test_local_meshes_reconstruct_posed_meshes(name, kwargs) -> None:
    model = create_model(name, **kwargs)
    frames = np.asarray(model.forward_skeleton(**model.get_rest_pose()))
    for meshes in (model.visual, model.collision):
        vertices, faces, offset = [], [], 0
        for mesh, body in zip(meshes.meshes, meshes.body_indices, strict=True):
            vertices.append(np.asarray(mesh.vertices) @ frames[body, :3, :3].T + frames[body, :3, 3])
            faces.append(np.asarray(mesh.faces) + offset)
            offset += len(mesh.vertices)
        expected = meshes.pose(frames)[0]
        np.testing.assert_allclose(np.concatenate(vertices), expected.vertices, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(np.concatenate(faces), expected.faces)


@pytest.mark.parametrize(("name", "kwargs"), model_cases.RUNTIME_CASES)
def test_arbitrary_leading_dimensions(name, kwargs) -> None:
    model = create_model(name, **kwargs)
    for batch_shape in LEADING_DIM_BATCH_SHAPES:
        frames = model.forward_skeleton(**model.get_rest_pose(batch_dims=batch_shape))
        assert frames.shape == (*batch_shape, model.num_bodies, 4, 4)
        for meshes in (model.visual, model.collision):
            assert len(meshes.pose(frames)) == int(np.prod(batch_shape))
