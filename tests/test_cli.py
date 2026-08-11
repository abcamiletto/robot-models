from pathlib import Path

import pytest
from typer.testing import CliRunner

from robot_models import _cli
from robot_models.brainco import _io as brainco_io


@pytest.mark.fast
def test_download_accepts_an_explicit_output_directory(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "brainco"
    calls = {}

    def download_model(*, output_dir: Path) -> Path:
        calls["download"] = output_dir
        return output_dir

    def set_model_path(model: str, path: Path) -> None:
        calls["config"] = (model, path)

    monkeypatch.setattr(brainco_io, "download_model", download_model)
    monkeypatch.setattr(_cli.config, "set_model_path", set_model_path)

    result = CliRunner().invoke(_cli.app, ["download", "brainco", "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert calls == {
        "download": output_dir,
        "config": ("brainco", output_dir),
    }
