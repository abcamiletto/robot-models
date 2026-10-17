import json
import zipfile

import pytest

from robot_models import _assets as assets


@pytest.mark.fast
def test_model_dir_uses_cache_without_downloading(tmp_path, monkeypatch) -> None:
    (tmp_path / "g1").mkdir()
    monkeypatch.setattr(assets.config, "get_model_path", lambda name: None)
    monkeypatch.setattr(assets, "get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(assets, "download", lambda name: pytest.fail("download should not run on a cache hit"))

    assert assets.model_dir("g1") == tmp_path / "g1"


@pytest.mark.fast
@pytest.mark.parametrize("version", [1, 2, 3])
def test_updated_collision_assets_refresh_only_outdated_cache(tmp_path, monkeypatch, version):
    cached = tmp_path / "barrett"
    cached.mkdir()
    (cached / "source.json").write_text(json.dumps({"asset_version": version}))
    monkeypatch.setattr(assets.config, "get_model_path", lambda name: None)
    monkeypatch.setattr(assets, "get_cache_dir", lambda: tmp_path)
    downloads = []

    def download(name):
        downloads.append(name)
        return cached

    monkeypatch.setattr(assets, "download", download)
    assert assets.model_dir("barrett") == cached
    assert downloads == (["barrett"] if version < 3 else [])


@pytest.mark.fast
@pytest.mark.parametrize("configured", [False, True])
def test_explicit_outdated_assets_fail_without_changing_files(tmp_path, monkeypatch, configured):
    monkeypatch.setattr(assets.config, "get_model_path", lambda name: tmp_path if configured else None)
    (tmp_path / "source.json").write_text('{"asset_version": 1}')
    with pytest.raises(ValueError, match="asset version 3"):
        assets.model_dir("barrett", None if configured else tmp_path)
    assert (tmp_path / "source.json").read_text() == '{"asset_version": 1}'


@pytest.mark.fast
def test_stale_download_does_not_replace_existing_assets(tmp_path):
    archive = tmp_path / "stale.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("source.json", json.dumps({"asset_version": 1}))
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "right.xml").write_text("existing hand")
    with pytest.raises(ValueError, match="version 2"):
        assets.extract_archive(archive, existing, minimum_version=2)
    assert (existing / "right.xml").read_text() == "existing hand"
