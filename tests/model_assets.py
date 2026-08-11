"""Shared test asset paths."""

from pathlib import Path

from robot_models import _config as config

ASSET_DIR = Path(__file__).parent / "assets" / "models_hub"

TEST_ASSET_PATHS = {
    "brainco": Path("brainco"),
    "g1": Path("g1"),
    "myofullbody": Path("myofullbody"),
}

CONFIG_KEYS = {
    "brainco": "brainco",
    "g1": "g1",
    "myofullbody": "myofullbody",
}

TEST_MODEL_FILES_BY_CONFIG_KEY = {key: TEST_ASSET_PATHS[name] for name, key in CONFIG_KEYS.items()}


def get_test_model_file_for_config_key(config_key: str) -> Path:
    return ASSET_DIR / TEST_MODEL_FILES_BY_CONFIG_KEY[config_key]


def get_model_file(model_name: str) -> Path:
    model_path = config.get_model_path(CONFIG_KEYS[model_name])
    if model_path is not None:
        return model_path
    return ASSET_DIR / TEST_ASSET_PATHS[model_name]
