"""pytest configuration for robot-models tests."""

from pathlib import Path

import pytest

from robot_models import _config as config

ASSET_DIR = Path(__file__).parent / "assets" / "models_hub"
TEST_MODEL_PATHS = {
    "brainco": ASSET_DIR / "brainco",
    "g1": ASSET_DIR / "g1",
    "myofullbody": ASSET_DIR / "myofullbody",
    "smpl-humanoid-humenv": ASSET_DIR / "smpl-humanoid" / "humenv.xml",
    "smpl-humanoid-phc": ASSET_DIR / "smpl-humanoid" / "phc.xml",
    "smpl-humanoid-smplsim": ASSET_DIR / "smpl-humanoid" / "smplsim.xml",
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
