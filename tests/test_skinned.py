"""Robots built from Body Models skins (MANO, SMPL) against Body Models itself."""

import numpy as np
import pytest
import trimesh

from robot_models import add_hand, create_model
from robot_models._common.mjcf import MUJOCO_TO_MODEL

VARIANTS = [("mano", {"side": side}) for side in ("right", "left")]
VARIANTS += [("smpl", {"gender": gender}) for gender in ("neutral", "male", "female")]
IDS = [f"{name}-{next(iter(kwargs.values()))}" for name, kwargs in VARIANTS]
SHAPE = np.random.default_rng(1).normal(0, 2, 10).astype(np.float32)


def random_pose(name, seed=2):
    count = 45 if name == "mano" else 69
    return np.random.default_rng(seed).normal(0, 0.4, count).astype(np.float32)


def reference(name, kwargs):
    from body_models import create_model as body_model

    if name == "mano":
        return body_model("mano", flat_hand_mean=True, flip_shapedirs=kwargs["side"] == "left", **kwargs)
    return body_model(name, **kwargs)


def random_parameters(reference, shape, count, seed=0):
    """Random Body Models parameters of one shape: every rotation and a translation."""
    rng = np.random.default_rng(seed)
    params = {"shape": np.tile(shape, (count, 1)).astype(np.float32)}
    for key, spec in reference.parameter_spec.items():
        if key != "shape":
            scale = 0.5 if key == "global_translation" else 0.6
            params[key] = rng.normal(0, scale, (count, *spec.dims)).astype(np.float32)
    return params


def native(reference, model):
    """Body Models' joint index of each of the model's joints."""
    return [reference.joint_names.index(name) for name in model.joint_names]


@pytest.mark.parametrize(("name", "kwargs"), VARIANTS, ids=IDS)
@pytest.mark.parametrize("shape", [None, SHAPE], ids=["mean", "shaped"])
def test_joint_frames_and_skin_match_body_models(name, kwargs, shape):
    model, body = create_model(name, shape=shape, **kwargs), reference(name, kwargs)
    params = random_parameters(body, np.zeros(10) if shape is None else shape, 6)
    ours = model.from_body_models(**params)
    expected = np.asarray(body.forward_skeleton(**params))[:, native(body, model)]
    np.testing.assert_allclose(model.forward_skeleton(**ours), expected, atol=2e-6)

    back = model.to_body_models(**ours)
    np.testing.assert_allclose(back["shape"], params["shape"])
    np.testing.assert_allclose(np.asarray(body.forward_skeleton(**back))[:, native(body, model)], expected, atol=2e-6)
    vertices = body.forward_vertices(**params)
    np.testing.assert_allclose(model.forward_vertices(**ours), vertices, atol=2e-6)
    selected = model.forward_vertices(**ours, vertex_indices=[5, 0, 5])
    np.testing.assert_allclose(selected, vertices[:, [5, 0, 5]], atol=2e-6)
    np.testing.assert_array_equal(model.skin_faces, body.faces)
    with pytest.raises(ValueError, match="shape"):
        model.from_body_models(**params | {"shape": params["shape"] + 1})


@pytest.mark.parametrize(("name", "kwargs"), VARIANTS, ids=IDS)
@pytest.mark.parametrize(
    ("shape", "conformed"), [(None, False), (SHAPE, False), (SHAPE, True)], ids=["mean", "shaped", "conformed"]
)
def test_links_are_the_skin_at_their_pose(name, kwargs, shape, conformed):
    """Posed at the pose its links are cut at (flat by default), every link vertex but the fan centers
    closing its openings is a skin vertex."""
    from scipy.spatial import cKDTree

    pose = random_pose(name) if conformed else None
    model = create_model(name, shape=shape, conform_to=pose, **kwargs)
    params = model.get_rest_pose() | ({} if pose is None else {model._POSE: pose})
    skin = cKDTree(model.forward_vertices(**params))
    distance, _ = skin.query(model.forward_meshes(**params)[0].vertices)
    offsets = model._geometry_assets("visual").vertex_offsets
    for link, start, end in zip(model.link_names, offsets[:-1], offsets[1:], strict=True):
        assert np.mean(distance[start:end] < 1e-6) > 0.85, link


@pytest.mark.parametrize(("name", "kwargs"), VARIANTS, ids=IDS)
@pytest.mark.parametrize("value", [-5.0, 5.0])
def test_extreme_shape_joint_positions_match_body_models(name, kwargs, value):
    shape = np.full(10, value)
    model, body = create_model(name, shape=shape, **kwargs), reference(name, kwargs)
    params = body.get_rest_pose(dtype=np.float64) | {"shape": shape}
    actual = model.forward_skeleton(**model.from_body_models(**params))
    expected = body.forward_skeleton(**params)[native(body, model)]
    np.testing.assert_allclose(actual[..., :3, 3], expected[..., :3, 3], atol=1e-6, rtol=0)


@pytest.mark.parametrize(("name", "kwargs"), VARIANTS, ids=IDS)
def test_collision_pieces_follow_the_links(name, kwargs):
    """At another shape and pose, the piece vertices keep their distances to their link's surface."""
    from scipy.spatial import cKDTree

    flat = create_model(name, conform_to=np.zeros_like(random_pose(name)), **kwargs)
    np.testing.assert_array_equal(
        flat.collision_meshes[0].vertices, create_model(name, **kwargs).collision_meshes[0].vertices
    )
    distances = []
    for shape, pose in ((None, None), (SHAPE, random_pose(name))):
        model = create_model(name, shape=shape, conform_to=pose, **kwargs)
        surfaces = [cKDTree(trimesh.sample.sample_surface(mesh, 50000, seed=0)[0]) for mesh in model.link_meshes]
        owners = [model.link_joint_indices.index(joint) for joint in model.collision_joint_indices]
        pieces = zip(model.collision_meshes, owners, strict=True)
        distances.append(np.concatenate([surfaces[owner].query(piece.vertices)[0] for piece, owner in pieces]))
    change = np.abs(distances[1] - distances[0])
    assert np.percentile(change, 99) < (0.003 if name == "mano" else 0.015)  # a body bends more than a hand


@pytest.mark.parametrize("side", ("right", "left"))
def test_mujoco_hand_matches_the_model_of_its_geometry(side):
    mujoco = pytest.importorskip("mujoco")
    rng = np.random.default_rng(3)
    for shape, conform_to in ((-2 * np.ones(10, np.float32), None), (SHAPE, random_pose("mano"))):
        hand = create_model("mano", side=side, shape=shape, conform_to=conform_to)
        scene = mujoco.MjSpec()
        _, root = add_hand(scene, "mano", side=side, free=True, shape=shape, conform_to=conform_to)
        model = scene.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        if conform_to is None:
            assert data.ncon == 0  # quiet at rest, whatever the shape
        volume = sum(mesh.volume for mesh in hand.link_meshes)
        np.testing.assert_allclose(model.body_subtreemass[model.body(root.name).id], 1100 * volume, rtol=1e-6)

        pose = rng.normal(0, 0.4, hand.num_dofs).astype(np.float32)
        data.qpos[:] = hand.to_qpos(pose)
        mujoco.mj_kinematics(model, data)
        bodies = [model.body(f"mano_{side}_{joint}").id for joint in hand.joint_names]
        positions = (data.xpos[bodies] - data.xpos[bodies[0]]) @ MUJOCO_TO_MODEL.T
        np.testing.assert_allclose(positions, hand.forward_skeleton(pose)[:, :3, 3], atol=1e-6)
        expected = hand.forward_meshes(geometry="collision", hand_pose=pose)[0].vertices
        world = []
        for geom in (geom for geom in range(model.ngeom) if model.geom_contype[geom]):
            mesh = model.geom_dataid[geom]
            local = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            world.append(local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
        world = (np.concatenate(world) - data.xpos[bodies[0]]) @ MUJOCO_TO_MODEL.T
        np.testing.assert_allclose(world.min(0), expected.min(0), atol=1e-6)
        np.testing.assert_allclose(world.max(0), expected.max(0), atol=1e-6)


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
def test_shaped_models_match_across_backends_and_skin_is_differentiable(name):
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    numpy_model = create_model(name, shape=SHAPE)
    rng = np.random.default_rng(6)
    params = numpy_model.get_rest_pose(batch_dims=(2, 3))
    params[numpy_model._POSE] = rng.normal(0, 0.3, params[numpy_model._POSE].shape).astype(np.float32)
    skeleton, vertices = numpy_model.forward_skeleton(**params), numpy_model.forward_vertices(**params)

    torch_model = create_model(name, runtime="torch", shape=SHAPE)
    torch_params = {key: torch.tensor(value, requires_grad=True) for key, value in params.items()}
    np.testing.assert_allclose(torch_model.forward_skeleton(**torch_params).detach().numpy(), skeleton, atol=1e-6)
    torch_vertices = torch_model.forward_vertices(**torch_params)
    np.testing.assert_allclose(torch_vertices.detach().numpy(), vertices, atol=1e-5)
    torch_vertices.square().sum().backward()
    for key, value in torch_params.items():
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0, key

    jax_model = create_model(name, runtime="jax", shape=SHAPE)
    jax_params = {key: jax.numpy.asarray(value) for key, value in params.items()}

    @jax.jit
    def roundtrip(model, params):
        return model.forward_skeleton(**model.from_body_models(**model.to_body_models(**params)))

    np.testing.assert_allclose(roundtrip(jax_model, jax_params), skeleton, atol=1e-6)

    def energy(pose):
        return jax.numpy.square(jax_model.forward_vertices(**(jax_params | {jax_model._POSE: pose}))).sum()

    gradient = jax.jit(jax.grad(energy))(jax_params[jax_model._POSE])
    np.testing.assert_allclose(gradient, torch_params[jax_model._POSE].grad.numpy(), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("side", ("right", "left"))
def test_mano_enforces_wide_limits_and_reports_anatomical_ones(side):
    mujoco = pytest.importorskip("mujoco")
    from robot_models import _assets

    hand = create_model("mano", side=side)
    model = mujoco.MjModel.from_xml_path(str(_assets.model_dir("mano") / f"{side}.xml"))
    assert model.jnt_limited.all()
    np.testing.assert_allclose(model.jnt_range, hand.actuated_joint_limits, atol=1e-6)
    lower, upper = np.asarray(hand.anatomical_limits).T
    enforced_lower, enforced_upper = np.asarray(hand.actuated_joint_limits).T
    assert np.all((enforced_lower < lower) & (upper < enforced_upper))
    for preset in ("flat", "rest", "open", "closed"):
        pose = hand.get_rest_pose(hands=preset)["hand_pose"]
        assert np.all((lower - 1e-6 <= pose) & (pose <= upper + 1e-6)), preset
