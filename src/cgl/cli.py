"""CGL experiment commands. Every execution emits a durable artifact directory."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from cgl.config import GenerationConfig, ModelConfig, TrainingConfig, load_config

app = typer.Typer(no_args_is_help=True, help="Causal Generalization Laboratory")
sources_app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
app.add_typer(sources_app, name="sources")
app.add_typer(data_app, name="data")


def root() -> Path:
    return Path(os.environ.get("CGL_ROOT", Path.cwd())).resolve()


@sources_app.command("lock")
def sources_lock():
    from cgl.sources import lock_sources

    typer.echo(json.dumps(lock_sources(root()), indent=2))


@sources_app.command("download")
def sources_download(model: Annotated[list[str] | None, typer.Option()] = None):
    from cgl.sources import DEFAULT_MODELS, download_models

    download_models(root(), model or DEFAULT_MODELS)


@data_app.command("originals")
def data_originals():
    from cgl.sources import prepare_originals

    typer.echo(json.dumps(prepare_originals(root()), indent=2))


@app.command()
def doctor():
    """Run P000 on the real primary model and released data."""
    from cgl.preflight import preflight

    typer.echo(preflight(root()))


@app.command()
def train(config: Path, resume: str | None = None):
    from cgl.training import train as execute

    typer.echo(execute(root(), load_config(config, TrainingConfig), resume=resume))


@app.command()
def generate(config: Path):
    from cgl.evaluation import generate as execute

    typer.echo(execute(root(), load_config(config, GenerationConfig)))


@app.command()
def judge(generations: Path, model: str = "Qwen/Qwen2.5-14B-Instruct", limit: int | None = None):
    from cgl.evaluation import judge as execute

    typer.echo(
        execute(root(), generations, ModelConfig(repo_id=model, quantization="nf4"), limit=limit)
    )


@app.command("score")
def score(
    panel: Path,
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter: str | None = None,
    basis: str | None = None,
    layer: int | None = None,
    dose: float = 1.0,
    operation: str = "ablate",
    limit: int | None = None,
):
    from cgl.evaluation import score_pairs

    typer.echo(
        score_pairs(
            root(),
            ModelConfig(repo_id=model),
            panel,
            adapter=adapter,
            basis_path=basis,
            layer=layer,
            operation=operation,
            dose=dose,
            limit=limit,
        )
    )


@app.command()
def status():
    """List recorded executions without inferring scientific confirmation."""
    records = []
    for path in sorted((root() / "runs").glob("*/*/manifest.json")):
        manifest = json.loads(path.read_text())
        records.append({key: manifest.get(key) for key in ("id", "experiment", "status", "error")})
    typer.echo(json.dumps(records, indent=2))


@app.command()
def power(effect: float = 0.15, run_sd: float = 0.1, draws: int = 200):
    from cgl.statistics import power_simulation

    typer.echo(json.dumps(power_simulation(effect=effect, run_sd=run_sd, draws=draws), indent=2))
