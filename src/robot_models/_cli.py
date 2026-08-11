"""Command-line configuration and asset management."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Annotated

import typer

from robot_models import _config as config
from robot_models._cache import get_cache_dir
from robot_models._catalog import DOWNLOAD_SPECS

app = typer.Typer(add_completion=False, no_args_is_help=False)

DOWNLOAD_NAMES = tuple(DOWNLOAD_SPECS)


@app.callback(invoke_without_command=True)
def show_config(ctx: typer.Context) -> None:
    """Configure and download robot-model assets."""
    if ctx.invoked_subcommand is not None:
        return
    typer.echo(f"Config file: {config.CONFIG_FILE}\n")
    typer.echo(f"Asset cache: {get_cache_dir()}\n")
    typer.echo("Current settings:")
    for model in config.ASSET_KEYS:
        typer.echo(f"  {model}: {config.get_model_path(model) or '(not set)'}")


@app.command()
def set(model: Annotated[str, typer.Argument()], path: Path) -> None:
    """Validate and save a model asset path."""
    _require_choice(model, config.ASSET_KEYS, "model asset")
    config.set_model_path(model, path)
    typer.echo(f"Set {model} = {config.get_model_path(model)}")


@app.command()
def unset(model: Annotated[str, typer.Argument()]) -> None:
    """Remove a model asset path from the config."""
    _require_choice(model, config.ASSET_KEYS, "model asset")
    config.unset_model_path(model)
    typer.echo(f"Removed {model} from config")


@app.command()
def download(
    model: Annotated[str, typer.Argument()] = "all",
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Exact destination for one model, or parent directory for all models."),
    ] = None,
) -> None:
    """Download one model family, or every supported family."""
    _require_choice(model, (*DOWNLOAD_NAMES, "all"), "download")
    names = DOWNLOAD_NAMES if model == "all" else (model,)
    for name in names:
        destination = output_dir / name if output_dir is not None and model == "all" else output_dir
        _download(name, destination)


def _download(name: str, output_dir: Path | None = None) -> None:
    spec = DOWNLOAD_SPECS[name]
    downloader = getattr(import_module(spec.module), spec.function)
    kwargs = {}
    if output_dir is not None:
        kwargs["output_dir"] = output_dir
    result = downloader(**kwargs)
    if spec.output_key is not None:
        _save_paths({spec.output_key: result})
    elif isinstance(result, Mapping):
        _save_paths(result)
    else:
        raise TypeError(f"{spec.module}.{spec.function} must return a mapping")


def _save_paths(paths: Mapping[str, str | Path]) -> None:
    for key, path in sorted(paths.items()):
        config.set_model_path(key, path)
        typer.echo(f"Set {key} = {path}")


def _require_choice(value: str, choices: tuple[str, ...], label: str) -> None:
    if value not in choices:
        expected = ", ".join(choices)
        raise typer.BadParameter(f"Unknown {label} {value!r}. Expected one of: {expected}")


def main() -> None:
    app()


__all__ = ["app", "main"]
