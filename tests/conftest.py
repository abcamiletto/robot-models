"""pytest configuration for robot-models tests."""

from pathlib import Path

import pytest

from robot_models import _config as config

ASSET_DIR = Path(__file__).parent / "assets" / "models_hub"


@pytest.fixture(autouse=True)
def setup_model_paths(monkeypatch):
    """Use configured model paths, then test assets."""
    get_config_model_path = config.get_model_path

    def get_model_path(model: str):
        configured = get_config_model_path(model)
        if configured is not None:
            return configured
        test_path = ASSET_DIR / model
        return test_path if test_path.exists() else None

    monkeypatch.setattr(config, "get_model_path", get_model_path)
