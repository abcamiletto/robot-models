"""pytest configuration for robot-models tests."""

from pathlib import Path

import model_cases
import pytest

from robot_models import _config as config
from robot_models._catalog import asset_spec

ASSET_DIR = Path(__file__).parent / "assets" / "models_hub"


@pytest.fixture(autouse=True)
def setup_model_paths(monkeypatch):
    """Use configured model paths, then test assets."""
    get_config_model_path = config.get_model_path

    def get_model_path(model: str, *, source: str | None = None):
        configured = get_config_model_path(model, source=source)
        if configured is not None:
            return configured
        test_path = ASSET_DIR / asset_spec(model, source).name
        return test_path if test_path.exists() else None

    monkeypatch.setattr(config, "get_model_path", get_model_path)


@pytest.fixture(autouse=True)
def unpublished_model_assets(request, setup_model_paths):
    """Unpublished source variants need locally prepared assets (never download lookalikes)."""
    parameters = getattr(getattr(request.node, "callspec", None), "params", {})
    if "model_class" not in parameters and "case" not in parameters:
        return
    name = parameters.get("name")
    if name is None and "case" in parameters:
        name = parameters["case"][0]
    family, options = model_cases.model_selection(name)
    source = options.get("source")
    if not asset_spec(family, source).downloadable and config.get_model_path(family, source=source) is None:
        pytest.skip(f"{name} requires a locally prepared source bundle; see docs/models/baselines.md")
