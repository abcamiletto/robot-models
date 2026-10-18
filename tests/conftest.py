"""pytest configuration for robot-models tests."""

from pathlib import Path

import pytest

from robot_models import _config as config
from robot_models._catalog import MODEL_SPECS

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


def body_models_files(name: str) -> bool:
    """Whether the user's Body Models files of a locally built model are configured."""
    spec = MODEL_SPECS[name]
    try:
        spec.source.path(spec.variants[0])
    except (FileNotFoundError, ImportError):
        return False
    return True


def pytest_collection_modifyitems(items):
    """Skip cases of models built from Body Models files, such as MANO, where none are configured."""
    built = {name for name, spec in MODEL_SPECS.items() if spec.source is not None}
    missing = {name for name in built if not body_models_files(name)}
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        values = [v for value in params.values() for v in (value if isinstance(value, tuple) else (value,))]
        if missing.intersection(v for v in values if isinstance(v, str)):
            item.add_marker(pytest.mark.skip(reason="the model's Body Models files are not configured"))
