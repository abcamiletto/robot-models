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
def test_download_all_skips_unpublished_shadow_sources(tmp_path, monkeypatch):
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
    assert not any(source is not None for _, source, _ in calls)
    assert _config.get_model_path("shadow") == output / "shadow"
    assert "shadow-dexgraspnet" not in _config.get_config().get("paths", {})


@pytest.mark.fast
def test_set_rejects_mismatched_shadow_source_before_saving(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(_config, "CONFIG_FILE", tmp_path / "config/config.toml")
    (tmp_path / "source.json").write_text(
        '{"upstream":"https://github.com/tengyu-liu/GenDexGrasp","robot":"shadowhand"}'
    )
    result = CliRunner().invoke(_cli.app, ["set", "shadow", str(tmp_path), "--source", "dexgraspnet"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "shadow-dexgraspnet" not in _config.get_config().get("paths", {})
    result = CliRunner().invoke(_cli.app, ["set", "shadow", str(tmp_path), "--source", "gendexgrasp"])
    assert result.exit_code == 0, result.output
    assert _config.get_model_path("shadow", source="gendexgrasp") == tmp_path


@pytest.mark.fast
@pytest.mark.parametrize("source", ["gendexgrasp", "dexgraspnet"])
def test_unpublished_shadow_download_explains_local_preparation(source):
    result = CliRunner().invoke(_cli.app, ["download", "shadow", "--source", source])
    assert result.exit_code != 0
    assert "not published" in result.output
    assert "prepare_baseline_assets.py" in result.output


@pytest.mark.fast
def test_source_requires_a_supported_model_selection():
    runner = CliRunner()
    result = runner.invoke(_cli.app, ["download", "all", "--source", "dexgraspnet"])
    assert result.exit_code != 0
    assert "single model" in result.output
    result = runner.invoke(_cli.app, ["download", "allegro", "--source", "dexgraspnet"])
    assert result.exit_code != 0
    assert "Unknown source" in result.output
