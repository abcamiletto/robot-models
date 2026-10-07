import numpy as np
import pytest

from robot_models._common.kinematics import affine_transforms, select_joint_chains

pytestmark = pytest.mark.fast


def test_affine_transforms_broadcasts_linear_and_translation_batches() -> None:
    linear = np.broadcast_to(np.eye(3), (3, 2, 3, 3))
    translation = np.arange(6).reshape(1, 2, 3)

    transforms = affine_transforms(linear, translation, xp=np)

    assert transforms.shape == (3, 2, 4, 4)
    np.testing.assert_array_equal(transforms[..., :3, 3], np.broadcast_to(translation, (3, 2, 3)))
    expected_bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (3, 2, 4))
    np.testing.assert_array_equal(transforms[..., 3, :], expected_bottom)


def test_select_joint_chains_returns_ancestor_closed_joints() -> None:
    parents = [-1, 0, 1, 2, 0, 4, 5, 0]

    assert select_joint_chains(parents, [6, 2]) == (0, 1, 2, 4, 5, 6)
    assert select_joint_chains(parents, [3, 3]) == (0, 1, 2, 3)
    assert select_joint_chains(parents, [0]) == (0,)
    assert select_joint_chains(parents, []) == ()
    with pytest.raises(IndexError, match="joint_indices"):
        select_joint_chains(parents, [len(parents)])
