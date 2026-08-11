import pytest

import robot_models.brainco._io as brainco_io
from robot_models import _config as config


@pytest.mark.fast
def test_brainco_get_model_path_uses_cache_without_downloading(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "brainco"
    (cache_dir / "meshes" / "left").mkdir(parents=True)
    (cache_dir / "meshes" / "right").mkdir(parents=True)
    (cache_dir / "left.xml").touch()
    (cache_dir / "right.xml").touch()

    monkeypatch.setattr(brainco_io.config, "get_model_path", lambda name: None)
    monkeypatch.setattr(brainco_io, "get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        brainco_io,
        "download_hf_archive",
        lambda *args, **kwargs: pytest.fail("download should not run on a cache hit"),
    )

    assert brainco_io.get_model_path() == cache_dir


@pytest.mark.fast
def test_validate_model_path_myofullbody(tmp_path) -> None:
    xml_path = tmp_path / "body" / "myofullbody.xml"
    xml_path.parent.mkdir(parents=True)
    xml_path.touch()

    assert config.validate_model_path("myofullbody", tmp_path) == tmp_path


@pytest.mark.fast
def test_g1_get_model_path_uses_cache(tmp_path, monkeypatch) -> None:
    from robot_models.g1 import _io as g1_io

    monkeypatch.setattr(g1_io, "get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(g1_io.config, "get_model_path", lambda model: None)
    monkeypatch.setattr(
        g1_io,
        "download_hf_archive",
        lambda *args, **kwargs: pytest.fail("download should not run on a cache hit"),
    )

    cache_xml = tmp_path / "g1" / "g1.xml"
    cache_xml.parent.mkdir(parents=True)
    cache_xml.touch()
    assert g1_io.get_model_path() == cache_xml
