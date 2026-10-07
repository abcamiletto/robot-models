"""pytest configuration for robot-models tests."""

from pathlib import Path

import pytest

from robot_models import _config as config

ASSET_DIR = Path(__file__).parent / "assets" / "models_hub"
TEST_MODEL_PATHS = {
    "allegro": ASSET_DIR / "allegro",
    "brainco": ASSET_DIR / "brainco",
    "g1": ASSET_DIR / "g1",
    "h1": ASSET_DIR / "h1",
    "t1": ASSET_DIR / "t1",
    "shadow": ASSET_DIR / "shadow",
}


@pytest.fixture(autouse=True)
def setup_model_paths(monkeypatch):
    """Use configured model paths, then test assets."""
    get_config_model_path = config.get_model_path

    def get_model_path(model: str):
        model_path = get_config_model_path(model)
        if model_path is not None:
            return model_path

        test_path = TEST_MODEL_PATHS.get(model)
        return test_path if test_path is not None and test_path.exists() else None

    monkeypatch.setattr(config, "get_model_path", get_model_path)
