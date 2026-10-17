"""One Shadow family, isolated source assets and source-preserving model state."""

import pickle
from pathlib import Path

import model_cases
import numpy as np
import pytest

from robot_models import _assets as assets
from robot_models import _config as config
from robot_models import create_model, list_models
from robot_models.shadow.numpy import ShadowHand

SHADOW_CASES = [case for case in model_cases.MODELS if case[0] == "shadow" or case[0] in model_cases.SHADOW_SOURCES]


@pytest.mark.fast
def test_one_shadow_family_with_explicit_sources():
    assert list_models(pattern="shadow*") == ["shadow"]
    assert ShadowHand.available_sources == ("menagerie", "gendexgrasp", "dexgraspnet")
    for source in ("unknown", None):
        with pytest.raises(ValueError, match="Unknown source"):
            ShadowHand(source=source)
    for source in ("gendexgrasp", "dexgraspnet"):
        with pytest.raises(ValueError, match="side"):
            ShadowHand(source=source, side="left")


@pytest.mark.fast
def test_source_paths_do_not_overwrite_default_or_other_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config/config.toml")
    for source in ShadowHand.available_sources:
        directory = tmp_path / source
        directory.mkdir()
        config.set_model_path("shadow", directory, source=source)
    assert config.get_model_path("shadow") == tmp_path / "menagerie"
    for source in ShadowHand.available_sources:
        assert assets.model_dir("shadow", source=source) == tmp_path / source
    config.unset_model_path("shadow", source="gendexgrasp")
    assert config.get_config()["paths"] == {
        "shadow": str(tmp_path / "menagerie"),
        "shadow-dexgraspnet": str(tmp_path / "dexgraspnet"),
    }


@pytest.mark.parametrize(("name", "model_class", "kwargs"), SHADOW_CASES)
def test_sources_use_one_class_and_resolve_common_anatomy(name, model_class, kwargs):
    hand = create_model("shadow", **kwargs)
    assert type(hand) is ShadowHand
    assert hand.source == kwargs.get("source", "menagerie")
    assert hand.num_dofs == (22 if hand.source == "dexgraspnet" else 24)
    assert len(hand.common_joints) == 16
    for joint in hand.common_joints:
        hand.joint_index(joint)
    if hand.source == "menagerie":
        poses = Path(__file__).parents[1] / "src/robot_models/shadow/assets/poses.npz"
        with np.load(poses) as original:
            for preset in ("flat", "rest"):
                np.testing.assert_array_equal(
                    hand.get_rest_pose(hands=preset)["hand_pose"], original[f"right_{preset}"]
                )


@pytest.mark.parametrize(("name", "model_class", "kwargs"), SHADOW_CASES)
def test_torch_pickle_preserves_shadow_source(name, model_class, kwargs):
    pytest.importorskip("torch")
    hand = model_cases.backend_model_class(name, "torch")(**kwargs)
    restored = pickle.loads(pickle.dumps(hand))
    assert type(restored) is type(hand)
    assert restored.source == hand.source
    assert restored.side == hand.side
    assert restored.common_joints == hand.common_joints
    expected = hand.forward_skeleton(**hand.get_rest_pose(hands="flat"))
    actual = restored.forward_skeleton(**restored.get_rest_pose(hands="flat"))
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
