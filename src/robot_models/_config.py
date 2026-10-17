"""Persistent model-asset path configuration."""

import json
import tomllib
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

from robot_models._catalog import MODEL_SPECS

CONFIG_DIR = Path(user_config_dir("robot-models"))
CONFIG_FILE = CONFIG_DIR / "config.toml"

Config = dict[str, Any]


def get_config() -> Config:
    """Read the user configuration, returning an empty mapping when absent."""
    if not CONFIG_FILE.exists():
        return {}
    return tomllib.loads(CONFIG_FILE.read_text())


def get_model_path(model: str) -> Path | None:
    """Return the configured asset directory for a model, if present."""
    path = get_config().get("paths", {}).get(model)
    return Path(path) if path else None


def set_model_path(model: str, path: str | Path) -> None:
    """Validate and store a model asset directory."""
    if model not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model!r}")
    config = get_config()
    config.setdefault("paths", {})[model] = str(validate_model_path(path))
    _write_config(config)


def unset_model_path(model: str) -> None:
    """Remove a model asset path if it is configured."""
    config = get_config()
    if "paths" in config and model in config["paths"]:
        del config["paths"][model]
        if not config["paths"]:
            del config["paths"]
        _write_config(config)


def validate_model_path(path: str | Path) -> Path:
    """Return ``path`` as an absolute asset directory, failing if it does not exist."""
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model asset directory not found: {path}")
    return path


def _write_config(config: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    if config.get("paths"):
        lines.append("[paths]")
        for model, path in sorted(config["paths"].items()):
            lines.append(f"{model} = {json.dumps(path)}")
    CONFIG_FILE.write_text("\n".join(lines) + "\n" if lines else "")
