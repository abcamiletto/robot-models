import model_cases
import numpy as np
import pytest
from trimesh import Trimesh

LEADING_DIM_BATCH_SHAPES = [(), (2,), (2, 2)]


def mesh_vertices(meshes):
    return np.stack([np.asarray(mesh.vertices) for mesh in meshes], axis=0)


def assert_pose_helpers_round_trip(model, pose) -> None:
    pose_by_joint = model.unpack_pose(pose)
    assert list(pose_by_joint) == list(dict.fromkeys(model.actuated_joint_names))
    np.testing.assert_array_equal(np.asarray(model.pack_pose(pose_by_joint)), np.asarray(pose))


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_meshes_match_across_backends(name, model_class, kwargs) -> None:
    numpy_model = model_class(**kwargs)
    numpy_params = numpy_model.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    expected = mesh_vertices(numpy_model.forward_meshes(**numpy_params))

    torch = pytest.importorskip("torch")
    torch_model = model_cases.backend_model_class(name, "torch")(**kwargs)
    torch_params = torch_model.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    with torch.no_grad():
        torch_meshes = torch_model.forward_meshes(**torch_params)

    pytest.importorskip("jax")
    import jax.numpy as jnp

    jax_model = model_cases.backend_model_class(name, "jax")(**kwargs)
    jax_params = jax_model.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    jax_meshes = jax_model.forward_meshes(**jax_params)

    assert all(isinstance(mesh, Trimesh) for mesh in (*torch_meshes, *jax_meshes))
    np.testing.assert_allclose(mesh_vertices(torch_meshes), expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(mesh_vertices(jax_meshes), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_rigid_body_contract(name, model_class, kwargs) -> None:
    model = model_class(**kwargs)
    params = model.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    pose_name = "hand_pose" if "hand_pose" in params else "body_pose"
    skeleton = model.forward_skeleton(**params)

    assert skeleton.shape[-3] == model.num_joints == len(model.joint_names)
    assert len(model.actuated_joint_names) == model.num_dofs
    assert len(model.actuated_joint_types) == model.num_dofs
    assert model.actuated_joint_limits.shape == (model.num_dofs, 2)
    assert params[pose_name].shape == (2, model.num_dofs)
    assert_pose_helpers_round_trip(model, params[pose_name])

    qpos = model.to_qpos(
        params[pose_name],
        global_rotation=params["global_rotation"],
        global_translation=params["global_translation"],
    )
    assert qpos.shape == (2, 7 + model.num_dofs)


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_link_meshes_reconstruct_forward_mesh(name, model_class, kwargs) -> None:
    model = model_class(**kwargs)
    params = model.get_rest_pose()
    transforms = np.asarray(model.forward_links(**params))

    vertices = []
    faces = []
    vertex_offset = 0
    for mesh, transform in zip(model.link_meshes, transforms, strict=True):
        vertices.append(np.asarray(mesh.vertices) @ transform[:3, :3].T + transform[:3, 3])
        faces.append(np.asarray(mesh.faces) + vertex_offset)
        vertex_offset += len(mesh.vertices)

    expected = model.forward_meshes(**params)[0]
    np.testing.assert_allclose(np.concatenate(vertices), expected.vertices, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.concatenate(faces), expected.faces)


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_arbitrary_leading_dimensions(name, model_class, kwargs) -> None:
    model = model_class(**kwargs)
    joint_indices = list(range(min(8, model.num_joints)))
    for batch_shape in LEADING_DIM_BATCH_SHAPES:
        params = model.get_rest_pose(batch_dims=batch_shape)
        links = model.forward_links(**params)
        skeleton = model.forward_skeleton(**params, joint_indices=joint_indices)

        assert links.shape == (*batch_shape, len(model.link_names), 4, 4)
        assert skeleton.shape == (*batch_shape, len(joint_indices), 4, 4)
