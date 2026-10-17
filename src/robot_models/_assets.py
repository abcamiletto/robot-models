"""Model asset directories: configured paths, the local cache, and Hugging Face downloads."""

import tarfile
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

from huggingface_hub import hf_hub_download
from platformdirs import user_cache_dir

from robot_models import _config as config

HF_MODEL_REPO_ID = "abcamiletto/robot-models"


def model_dir(name: str, model_path: Path | str | None = None) -> Path:
    """Resolve a model's asset directory from an explicit path, the config, the cache, or a download."""
    model_path = model_path if model_path is not None else config.get_model_path(name)
    if model_path is not None:
        return config.validate_model_path(model_path)
    cached = get_cache_dir() / name
    return cached if cached.is_dir() else download(name)


def download(name: str, output_dir: Path | None = None) -> Path:
    """Download and extract one model's ``assets.zip`` from Hugging Face."""
    output_dir = output_dir if output_dir is not None else get_cache_dir() / name
    print(f"Downloading {name} model to {output_dir}...")
    archive = hf_hub_download(HF_MODEL_REPO_ID, f"{name}/assets.zip", cache_dir=get_cache_dir() / "huggingface")
    extract_archive(Path(archive), output_dir)
    return output_dir


def get_cache_dir() -> Path:
    """Get the robot-models cache directory."""
    return Path(user_cache_dir("robot-models"))


def extract_archive(archive_path: Path, dest: Path) -> None:
    """Extract an archive completely before replacing its destination."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dest.name}-", dir=dest.parent) as temporary:
        temporary_dir = Path(temporary)
        contents = temporary_dir / "contents"
        contents.mkdir()

        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                _validate_paths(archive.namelist())
                archive.extractall(contents)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as archive:
                members = archive.getmembers()
                _validate_paths(member.name for member in members)
                archive.extractall(contents, members=members, filter="data")
        else:
            raise ValueError(f"Unsupported archive: {archive_path}")

        previous = temporary_dir / "previous"
        if dest.exists():
            dest.rename(previous)
        try:
            contents.rename(dest)
        except OSError:
            if previous.exists():
                previous.rename(dest)
            raise


def _validate_paths(names: Iterable[str]) -> None:
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe path in archive: {name}")


__all__ = ["HF_MODEL_REPO_ID", "download", "extract_archive", "get_cache_dir", "model_dir"]
