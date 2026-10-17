from pathlib import Path

import pytest
from typer.testing import CliRunner

from robot_models import _cli, _config


@pytest.mark.fast
def test_download_accepts_an_explicit_output_directory(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "brainco"
    calls = {}

    def download(name: str, output_dir: Path, *, source: str | None = None) -> Path:
        calls["download"] = (name, output_dir)
        return output_dir

    def set_model_path(model: str, path: Path, *, source: str | None = None) -> None:
        calls["config"] = (model, path)

    monkeypatch.setattr(_cli.assets, "download", download)
    monkeypatch.setattr(_cli.config, "set_model_path", set_model_path)

    result = CliRunner().invoke(_cli.app, ["download", "brainco", "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert calls == {"download": ("brainco", output_dir), "config": ("brainco", output_dir)}


@pytest.mark.fast
def test_download_all_keeps_shadow_source_paths_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(_config, "CONFIG_FILE", tmp_path / "config/config.toml")
    calls = []

    def download(name, output_dir, *, source=None):
        calls.append((name, source, output_dir))
        output_dir.mkdir(parents=True)
        return output_dir

    monkeypatch.setattr(_cli.assets, "download", download)
    output = tmp_path / "models"
    result = CliRunner().invoke(_cli.app, ["download", "all", "--output-dir", str(output)])
    assert result.exit_code == 0, result.output
    assert ("shadow", None, output / "shadow") in calls
    assert ("shadow", "dexgraspnet", output / "shadow-dexgraspnet") in calls
    assert not any(source == "gendexgrasp" for _, source, _ in calls)
    assert _config.get_model_path("shadow") == output / "shadow"
    assert _config.get_model_path("shadow", source="dexgraspnet") == output / "shadow-dexgraspnet"


@pytest.mark.fast
def test_source_requires_a_supported_model_selection():
    runner = CliRunner()
    result = runner.invoke(_cli.app, ["download", "all", "--source", "dexgraspnet"])
    assert result.exit_code != 0
    assert "single model" in result.output
    result = runner.invoke(_cli.app, ["download", "allegro", "--source", "dexgraspnet"])
    assert result.exit_code != 0
    assert "Unknown source" in result.output
