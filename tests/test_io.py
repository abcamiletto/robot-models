import pytest

from robot_models import _assets as assets


@pytest.mark.fast
def test_model_dir_uses_cache_without_downloading(tmp_path, monkeypatch) -> None:
    (tmp_path / "g1").mkdir()
    monkeypatch.setattr(assets.config, "get_model_path", lambda name: None)
    monkeypatch.setattr(assets, "get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(assets, "download", lambda name: pytest.fail("download should not run on a cache hit"))

    assert assets.model_dir("g1") == tmp_path / "g1"
