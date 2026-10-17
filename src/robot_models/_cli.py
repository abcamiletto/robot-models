"""Command-line configuration and asset management."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from robot_models import _assets as assets
from robot_models import _config as config
from robot_models._catalog import MODEL_SOURCES, MODEL_SPECS, asset_spec

app = typer.Typer(add_completion=False, no_args_is_help=False)

MODEL_NAMES = tuple(MODEL_SPECS)


@app.callback(invoke_without_command=True)
def show_config(ctx: typer.Context) -> None:
    """Configure and download robot-model assets."""
    if ctx.invoked_subcommand is not None:
        return
    typer.echo(f"Config file: {config.CONFIG_FILE}\n")
    typer.echo(f"Asset cache: {assets.get_cache_dir()}\n")
    typer.echo("Current settings:")
    for model in MODEL_NAMES:
        typer.echo(f"  {model}: {config.get_model_path(model) or '(not set)'}")
        for source, spec in MODEL_SOURCES.get(model, {}).items():
            if spec.name != model:
                path = config.get_model_path(model, source=source)
                typer.echo(f"    --source {source}: {path or '(not set)'}")


@app.command()
def set(
    model: Annotated[str, typer.Argument()],
    path: Path,
    source: Annotated[str | None, typer.Option(help="Named description source (Shadow only).")] = None,
) -> None:
    """Validate and save a model asset path."""
    _require_choice(model, MODEL_NAMES, "model")
    _require_source(model, source)
    config.set_model_path(model, path, source=source)
    typer.echo(f"Set {model} ({source or 'default'}) = {config.get_model_path(model, source=source)}")


@app.command()
def unset(
    model: Annotated[str, typer.Argument()],
    source: Annotated[str | None, typer.Option(help="Named description source (Shadow only).")] = None,
) -> None:
    """Remove a model asset path from the config."""
    _require_choice(model, MODEL_NAMES, "model")
    _require_source(model, source)
    config.unset_model_path(model, source=source)
    typer.echo(f"Removed {model} ({source or 'default'}) from config")


@app.command()
def download(
    model: Annotated[str, typer.Argument()] = "all",
    source: Annotated[str | None, typer.Option(help="Named description source (Shadow only).")] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Exact destination for one model, or parent directory for all models."),
    ] = None,
) -> None:
    """Download one model/source, or every published model/source bundle."""
    _require_choice(model, (*MODEL_NAMES, "all"), "model")
    if model == "all":
        if source is not None:
            raise typer.BadParameter("--source requires a single model")
        selections = [(name, None) for name, spec in MODEL_SPECS.items() if spec.downloadable]
        selections += [
            (name, source)
            for name, sources in MODEL_SOURCES.items()
            for source, spec in sources.items()
            if spec.downloadable and spec.name != name
        ]
    else:
        _require_source(model, source)
        selections = [(model, source)]
    for name, selected_source in selections:
        spec = asset_spec(name, selected_source)
        destination = output_dir / spec.name if output_dir is not None and model == "all" else output_dir
        try:
            path = assets.download(name, destination, source=selected_source)
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc)) from exc
        config.set_model_path(name, path, source=selected_source)
        typer.echo(f"Set {name} ({selected_source or 'default'}) = {path}")


def _require_source(model: str, source: str | None) -> None:
    if source is not None:
        _require_choice(source, tuple(MODEL_SOURCES.get(model, {})), "source")


def _require_choice(value: str, choices: tuple[str, ...], label: str) -> None:
    if value not in choices:
        expected = ", ".join(choices)
        raise typer.BadParameter(f"Unknown {label} {value!r}. Expected one of: {expected}")


def main() -> None:
    app()


__all__ = ["app", "main"]
