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
campaign_app = typer.Typer(no_args_is_help=True)
review_app = typer.Typer(no_args_is_help=True)
app.add_typer(sources_app, name="sources")
app.add_typer(data_app, name="data")
app.add_typer(campaign_app, name="campaign")
app.add_typer(review_app, name="review")


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


@app.command("execute")
def execute_operator(action: str, arguments: Path):
    """Execute any registered research operator with explicit YAML arguments."""
    import yaml

    from cgl.campaigns import ACTIONS
    from cgl.worker import execute

    if action not in ACTIONS:
        raise typer.BadParameter(f"Choose one of: {', '.join(sorted(ACTIONS))}")
    typer.echo(execute(root(), action, yaml.safe_load(arguments.read_text()) or {}))


@campaign_app.command("build")
def campaign_build(family: str, output: Path, arguments: Path | None = None):
    import yaml

    from cgl import recipes
    from cgl.artifacts import write_json

    if family not in recipes.FAMILIES:
        raise typer.BadParameter(f"Choose one of: {', '.join(recipes.FAMILIES)}")
    kwargs = yaml.safe_load(arguments.read_text()) if arguments else {}
    campaign = recipes.FAMILIES[family](**kwargs)
    write_json(output, campaign.model_dump(), exclusive=True)
    typer.echo(f"Wrote {len(campaign.jobs)} jobs to {output}")


@campaign_app.command("validate")
def campaign_validate(definition: Path):
    from cgl.campaigns import load_campaign

    campaign = load_campaign(definition)
    typer.echo(f"{campaign.id}: {len(campaign.jobs)} valid jobs; scope={campaign.scope}")


@campaign_app.command("run")
def campaign_run(definition: Path, keep_going: bool = True, retry_failed: bool = False):
    from cgl.campaigns import load_campaign, run_campaign

    state_path = run_campaign(
        root(), load_campaign(definition), keep_going=keep_going, retry_failed=retry_failed
    )
    typer.echo(state_path)
    if not json.loads(state_path.read_text())["complete"]:
        raise typer.Exit(1)


@review_app.command("generations")
def review_generations(inputs: Path, output: Path, per_condition: int = 12, seed: int = 0):
    from cgl.human import build_generation_review

    paths = {k: root() / v for k, v in json.loads(inputs.read_text()).items()}
    typer.echo(build_generation_review(paths, output, per_condition=per_condition, seed=seed))


@review_app.command("fidelity")
def review_fidelity(originals: Path, transformed: Path, output: Path, count: int = 64):
    from cgl.human import build_fidelity_review

    typer.echo(build_fidelity_review(originals, transformed, output, count=count))


@review_app.command("finalize")
def review_finalize(directory: Path, fidelity: bool = False):
    from cgl.human import finalize_fidelity_review, finalize_generation_review

    finalize = finalize_fidelity_review if fidelity else finalize_generation_review
    typer.echo(json.dumps(finalize(directory), indent=2))


@review_app.command("calibrate")
def review_calibrate(directory: Path, scores: Path, output: Path):
    from cgl.measurement import calibrate_judge

    paths = {key: root() / value for key, value in json.loads(scores.read_text()).items()}
    typer.echo(json.dumps(calibrate_judge(directory, paths, output), indent=2))
