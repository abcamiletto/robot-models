"""Robots built from Body Models skins (MANO, SMPL) against Body Models itself."""

import numpy as np
import pytest

from robot_models import add_hand, create_model
from robot_models._common.mjcf import MUJOCO_TO_MODEL

VARIANTS = [("mano", {"side": side}) for side in ("right", "left")]
VARIANTS += [("smpl", {"gender": gender}) for gender in ("neutral", "male", "female")]
IDS = [f"{name}-{next(iter(kwargs.values()))}" for name, kwargs in VARIANTS]


def reference(name, kwargs):
    from body_models import create_model as body_model

    if name == "mano":
        return body_model("mano", flat_hand_mean=True, flip_shapedirs=kwargs["side"] == "left", **kwargs)
    return body_model(name, **kwargs)


def random_parameters(model, reference, count, seed=0):
    """Random Body Models parameters: every rotation, a translation and a shape."""
    rng = np.random.default_rng(seed)
    params = {}
    for key, spec in reference.parameter_spec.items():
        scale = {"shape": 2.0, "global_translation": 0.5}.get(key, 0.6)
        params[key] = rng.normal(0, scale, (count, *spec.dims)).astype(np.float32)
    return params


def native(reference, model):
    """Body Models' joint index of each of the model's joints."""
    return [reference.joint_names.index(name) for name in model.joint_names]


@pytest.mark.parametrize(("name", "kwargs"), VARIANTS, ids=IDS)
def test_joint_frames_and_skin_match_body_models(name, kwargs):
    model, body = create_model(name, **kwargs), reference(name, kwargs)
    params = random_parameters(model, body, 6)
    ours = model.from_body_models(**params)
    expected = np.asarray(body.forward_skeleton(**params))[:, native(body, model)]
    np.testing.assert_allclose(model.forward_skeleton(**ours), expected, atol=2e-6)
    mean = params | {"shape": np.zeros_like(params["shape"])}
    converted = {key: value for key, value in model.from_body_models(**mean).items() if key != "shape"}
    mean_expected = np.asarray(body.forward_skeleton(**mean))[:, native(body, model)]
    np.testing.assert_allclose(model.forward_skeleton(**converted), mean_expected, atol=2e-6)

    back = model.to_body_models(**ours)
    np.testing.assert_allclose(np.asarray(body.forward_skeleton(**back))[:, native(body, model)], expected, atol=2e-6)
    vertices = body.forward_vertices(**params)
    np.testing.assert_allclose(model.forward_vertices(**ours), vertices, atol=2e-6)
    selected = model.forward_vertices(**ours, vertex_indices=[5, 0, 5])
    np.testing.assert_allclose(selected, vertices[:, [5, 0, 5]], atol=2e-6)
    np.testing.assert_array_equal(model.skin_faces, body.faces)


@pytest.mark.parametrize(("name", "kwargs"), VARIANTS, ids=IDS)
def test_rest_links_are_the_rest_skin(name, kwargs):
    """At rest, every link vertex but the fan centers closing its openings is a skin vertex, for any shape."""
    from scipy.spatial import cKDTree

    model = create_model(name, **kwargs)
    offsets = model._geometry_assets("visual").vertex_offsets
    for shape in (np.zeros(model.num_shapes), np.random.default_rng(1).normal(0, 2, model.num_shapes)):
        params = model.get_rest_pose() | {"shape": shape.astype(np.float32)}
        skin = cKDTree(model.forward_vertices(**params))
        links = model.forward_meshes(**params)[0]
        np.testing.assert_array_equal(links.faces, model.faces)
        distance, _ = skin.query(links.vertices)
        for link, start, end in zip(model.link_names, offsets[:-1], offsets[1:], strict=True):
            assert np.mean(distance[start:end] < 1e-6) > 0.85, link


@pytest.mark.parametrize(("name", "kwargs"), VARIANTS, ids=IDS)
def test_extreme_shape_joint_positions_match_body_models(name, kwargs):
    model, body = create_model(name, **kwargs), reference(name, kwargs)
    params = body.get_rest_pose(batch_dims=(2,), dtype=np.float64)
    params["shape"] = np.array([[-5.0] * model.num_shapes, [5.0] * model.num_shapes])
    actual = model.forward_skeleton(**model.from_body_models(**params))
    expected = body.forward_skeleton(**params)[:, native(body, model)]
    np.testing.assert_allclose(actual[..., :3, 3], expected[..., :3, 3], atol=1e-6, rtol=0)


@pytest.mark.parametrize(("name", "kwargs"), VARIANTS, ids=IDS)
def test_collision_pieces_follow_the_shaped_links(name, kwargs):
    """Carried to another shape, the piece vertices keep their distances to their link's surface."""
    import trimesh
    from scipy.spatial import cKDTree

    model = create_model(name, **kwargs)
    visual, collision = model._geometry_assets("visual"), model._geometry_assets("collision")
    shapes = (np.zeros(model.num_shapes), np.random.default_rng(2).normal(0, 2, model.num_shapes))
    distances = []
    for shape in shapes:
        links = model._shaped(shape.astype(np.float32), visual.vertices, model._shape.visual_dirs)
        pieces = model._shaped(shape.astype(np.float32), collision.vertices, model._shape.collision_dirs)
        distance = np.zeros(len(pieces))
        for index, joint in enumerate(visual.joint_indices):
            start, end = visual.vertex_offsets[index : index + 2]
            faces = visual.faces[visual.face_offsets[index] : visual.face_offsets[index + 1]] - start
            surface, _ = trimesh.sample.sample_surface(trimesh.Trimesh(links[start:end], faces), 50000, seed=0)
            own = [piece for piece, owner in enumerate(collision.joint_indices) if owner == joint]
            mine = np.concatenate([np.arange(*collision.vertex_offsets[piece : piece + 2]) for piece in own])
            distance[mine] = cKDTree(surface).query(pieces[mine])[0]
        distances.append(distance)
    change = np.abs(distances[1] - distances[0])
    assert np.percentile(change, 99) < (0.003 if name == "mano" else 0.008)


@pytest.mark.parametrize("side", ("right", "left"))
def test_shaped_mujoco_hand_matches_forward_kinematics(side):
    import trimesh

    mujoco = pytest.importorskip("mujoco")
    hand = create_model("mano", side=side)
    rng = np.random.default_rng(3)
    for shape in (-2 * np.ones(10), rng.normal(0, 2, 10)):
        shape = shape.astype(np.float32)
        scene = mujoco.MjSpec()
        _, root = add_hand(scene, "mano", side=side, free=True, shape=shape)
        model = scene.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert data.ncon == 0  # quiet at rest, whatever the shape

        pose = rng.normal(0, 0.4, hand.num_dofs).astype(np.float32)
        data.qpos[:] = hand.to_qpos(pose)
        mujoco.mj_kinematics(model, data)
        bodies = [model.body(f"mano_{side}_{joint}").id for joint in hand.joint_names]
        positions = (data.xpos[bodies] - data.xpos[bodies[0]]) @ MUJOCO_TO_MODEL.T
        np.testing.assert_allclose(positions, hand.forward_skeleton(pose, shape=shape)[:, :3, 3], atol=1e-6)
        expected = hand.forward_meshes(geometry="collision", hand_pose=pose, shape=shape)[0].vertices
        geoms = [geom for geom in range(model.ngeom) if model.geom_contype[geom]]
        world = []
        for geom in geoms:
            mesh = model.geom_dataid[geom]
            local = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            world.append(local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
        world = (np.concatenate(world) - data.xpos[bodies[0]]) @ MUJOCO_TO_MODEL.T
        np.testing.assert_allclose(world.min(0), expected.min(0), atol=1e-6)
        np.testing.assert_allclose(world.max(0), expected.max(0), atol=1e-6)
        visual = hand._geometry_assets("visual")
        links = hand._shaped(shape, visual.vertices, hand._shape.visual_dirs)
        offsets, face_offsets = visual.vertex_offsets, visual.face_offsets
        spans = zip(offsets[:-1], offsets[1:], face_offsets[:-1], face_offsets[1:], strict=True)
        volume = sum(trimesh.Trimesh(links[a:b], visual.faces[c:d] - a).volume for a, b, c, d in spans)
        np.testing.assert_allclose(model.body_subtreemass[model.body(root.name).id], 1100 * volume, rtol=1e-6)


@pytest.mark.parametrize("name", ("mano",))
def test_shape_is_differentiable_in_torch_and_jax(name):
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    numpy_model = create_model(name)
    params = numpy_model.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    params["shape"] = np.random.default_rng(4).normal(0, 1, (2, 10)).astype(np.float32)
    params["hand_pose"] = numpy_model.get_rest_pose(batch_dims=(2,), hands="rest")["hand_pose"]
    expected = numpy_model.forward_skeleton(**params)

    torch_model = create_model(name, runtime="torch")
    torch_params = {key: torch.tensor(value, requires_grad=key == "shape") for key, value in params.items()}
    skeleton = torch_model.forward_skeleton(**torch_params)
    np.testing.assert_allclose(skeleton.detach().numpy(), expected, atol=1e-6)
    skeleton[..., :3, 3].sum().backward()
    assert torch.isfinite(torch_params["shape"].grad).all() and torch_params["shape"].grad.abs().sum() > 0

    jax_model = create_model(name, runtime="jax")
    jax_params = {key: jax.numpy.asarray(value) for key, value in params.items()}

    @jax.jit
    def tips(model, shape):
        return model.forward_skeleton(**(jax_params | {"shape": shape}))[..., :3, 3].sum()

    np.testing.assert_allclose(jax_model.forward_skeleton(**jax_params), expected, atol=1e-6)
    gradient = jax.grad(tips, argnums=1)(jax_model, jax_params["shape"])
    np.testing.assert_allclose(gradient, torch_params["shape"].grad.numpy(), rtol=1e-4, atol=1e-6)
    back = jax_model.from_body_models(**jax_model.to_body_models(**jax_params))
    np.testing.assert_allclose(back["hand_pose"], params["hand_pose"], atol=1e-5)


@pytest.mark.parametrize(("name", "kwargs"), [VARIANTS[0], VARIANTS[2]], ids=[IDS[0], IDS[2]])
@pytest.mark.parametrize(("pose_batch", "shape_batch"), [((), (2, 3)), ((2, 3), ()), ((2, 1), (1, 3))])
def test_shape_and_pose_batches_broadcast(name, kwargs, pose_batch, shape_batch):
    model = create_model(name, **kwargs)
    rng = np.random.default_rng(5)
    pose = rng.normal(0, 0.3, (*pose_batch, model.num_dofs)).astype(np.float32)
    shape = rng.normal(size=(*shape_batch, model.num_shapes)).astype(np.float32)
    batch = np.broadcast_shapes(pose_batch, shape_batch)
    params = {model._POSE: pose, "shape": shape}
    expanded = {
        model._POSE: np.broadcast_to(pose, (*batch, model.num_dofs)),
        "shape": np.broadcast_to(shape, (*batch, model.num_shapes)),
    }
    expected = model.forward_skeleton(**expanded)
    np.testing.assert_allclose(model.forward_skeleton(**params), expected, atol=1e-7)
    for indices in ([], [0], [3, 0, 3]):
        np.testing.assert_allclose(
            model.forward_skeleton(**params, joint_indices=indices), expected[..., indices, :, :], atol=1e-7
        )
    back = model.from_body_models(**model.to_body_models(**params))
    np.testing.assert_allclose(model.forward_skeleton(**back), expected, atol=1e-6)
    for geometry in ("visual", "collision"):
        actual = model.forward_meshes(**params, geometry=geometry)
        target = model.forward_meshes(**expanded, geometry=geometry)
        assert len(actual) == len(target) == np.prod(batch)
        for mesh, reference in zip(actual, target, strict=True):
            np.testing.assert_allclose(mesh.vertices, reference.vertices, atol=1e-7)


@pytest.mark.parametrize(("name", "kwargs"), VARIANTS, ids=IDS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_body_models_conversion_near_gimbal_lock(name, kwargs, dtype):
    model = create_model(name, **kwargs)
    pose = np.zeros((14, model.num_dofs), dtype=dtype)
    pose[:, 0], pose[:, 2] = 0.7, -0.4
    pose[:, 1] = np.repeat([np.pi / 2, -np.pi / 2], 7) + np.tile([0, 1e-9, -1e-9, 1e-7, -1e-7, 1e-4, -1e-4], 2)
    back = model.from_body_models(**model.to_body_models(**{model._POSE: pose}))
    np.testing.assert_allclose(model.forward_skeleton(**back), model.forward_skeleton(pose), atol=2e-6)


@pytest.mark.parametrize("name", ["mano", "smpl"])
def test_torch_shape_gradient_survives_dtype_conversion(name):
    torch = pytest.importorskip("torch")
    model = create_model(name, runtime="torch")
    pose = torch.zeros(model.num_dofs, dtype=torch.float64, requires_grad=True)
    shape = torch.ones(model.num_shapes, dtype=torch.float32, requires_grad=True)
    model.forward_skeleton(pose, shape=shape)[..., :3, 3].square().sum().backward()
    assert shape.grad is not None and torch.isfinite(shape.grad).all() and shape.grad.abs().sum() > 0


@pytest.mark.parametrize("name", ["mano", "smpl"])
def test_jax_conversion_with_two_batch_dimensions(name):
    jax = pytest.importorskip("jax")
    model = create_model(name, runtime="jax")
    params = model.get_rest_pose(batch_dims=(2, 3))
    params[model._POSE] = params[model._POSE] + 0.2
    params["shape"] = params["shape"] + 0.3

    @jax.jit
    def roundtrip(model, params):
        return model.forward_skeleton(**model.from_body_models(**model.to_body_models(**params)))

    np.testing.assert_allclose(roundtrip(model, params), model.forward_skeleton(**params), atol=1e-6)


@pytest.mark.parametrize("name", ["mano", "smpl"])
def test_skin_vertices_are_differentiable_in_torch_and_jax(name):
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    numpy_model = create_model(name)
    rng = np.random.default_rng(6)
    params = numpy_model.get_rest_pose(batch_dims=(2,))
    params[numpy_model._POSE] = rng.normal(0, 0.3, params[numpy_model._POSE].shape).astype(np.float32)
    params["shape"] = rng.normal(0, 1, (2, numpy_model.num_shapes)).astype(np.float32)
    expected = numpy_model.forward_vertices(**params)

    torch_model = create_model(name, runtime="torch")
    torch_params = {key: torch.tensor(value, requires_grad=True) for key, value in params.items()}
    vertices = torch_model.forward_vertices(**torch_params)
    np.testing.assert_allclose(vertices.detach().numpy(), expected, atol=1e-5)
    vertices.square().sum().backward()
    for key in (numpy_model._POSE, "shape", "global_rotation", "global_translation"):
        gradient = torch_params[key].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, key

    jax_model = create_model(name, runtime="jax")
    jax_params = {key: jax.numpy.asarray(value) for key, value in params.items()}

    @jax.jit
    def energy(model, shape):
        return model.forward_vertices(**(jax_params | {"shape": shape})).__pow__(2).sum()

    gradient = jax.grad(energy, argnums=1)(jax_model, jax_params["shape"])
    np.testing.assert_allclose(gradient, torch_params["shape"].grad.numpy(), rtol=1e-3, atol=1e-5)
