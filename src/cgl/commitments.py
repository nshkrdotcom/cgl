"""Public prediction commitments and outcome records linked to real resumed runs."""

import json
import subprocess
from pathlib import Path

from cgl.artifacts import (
    digest,
    file_hash,
    read_jsonl,
    utc_now,
    write_json,
    write_jsonl,
)


def publish_forecast(root: Path, predictions: Path):
    record = json.loads(predictions.read_text())
    content = {key: value for key, value in record.items() if key != "commitment_sha256"}
    if digest(content) != record["commitment_sha256"]:
        raise ValueError("Prediction commitment was altered")
    destination = root / "evidence/forecasts" / f"{record['commitment_sha256']}.json"
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()
    if not branch:
        raise ValueError("Publishing requires an attached Git branch")
    if destination.exists():
        if json.loads(destination.read_text()) != record:
            raise ValueError("Existing public prediction differs from this commitment")
    else:
        write_json(destination, record, exclusive=True)
    relative = str(destination.relative_to(root))
    subprocess.run(["git", "add", "--", relative], cwd=root, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", relative], cwd=root)
    if changed.returncode == 1:
        subprocess.run(
            [
                "git",
                "commit",
                "--only",
                "-m",
                f"Seal prospective forecast {record['commitment_sha256'][:12]}",
                "--",
                relative,
            ],
            cwd=root,
            check=True,
        )
    elif changed.returncode != 0:
        raise RuntimeError("Unable to inspect prediction commit status")
    revision = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative], cwd=root, text=True
    ).strip()
    subprocess.run(["git", "push", "origin", branch], cwd=root, check=True)
    receipt = root / "artifacts/forecast_receipts" / f"{record['commitment_sha256']}.json"
    if not receipt.exists():
        write_json(
            receipt,
            {
                "forecast_sha256": file_hash(destination),
                "git_commit": revision,
                "published_at": utc_now(),
                "public_file": relative,
            },
            exclusive=True,
        )
    return receipt


def outcome_record(
    features: Path,
    training_run: Path,
    evaluation: Path,
    output: Path,
    *,
    metric="mean_pcps",
    generation: Path | None = None,
):
    from cgl.verification import verify_run

    verify_run(training_run, experiment="training")
    verify_run(evaluation)
    early = read_jsonl(features)
    if len(early) != 1:
        raise ValueError("An outcome links one early checkpoint to one completed continuation")
    training = json.loads((training_run / "manifest.json").read_text())
    if training.get("resume_from", {}).get("parent_run_id") != early[0]["run_id"]:
        raise ValueError("Final training is not the recorded early checkpoint continuation")
    metrics = json.loads((training_run / "metrics.json").read_text())
    if not metrics.get("training_complete"):
        raise ValueError("Cannot score an unfinished continuation as the final target")
    summary = json.loads((evaluation / "summary.json").read_text())
    manifest = json.loads((evaluation / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Final evaluation did not complete")
    evaluated_model = manifest
    if metric == "misalignment_rate":
        if (
            generation is None
            or file_hash(generation / "generations.jsonl")
            != manifest["inputs"]["generations_sha256"]
        ):
            raise ValueError("Judge outcome needs its matching raw generation artifact")
        evaluated_model = json.loads((generation / "manifest.json").read_text())
        verify_run(generation, experiment="generation")
    if Path(evaluated_model["config"]["adapter"]).resolve() != (training_run / "adapter").resolve():
        raise ValueError("Outcome evaluated a different adapter from the completed continuation")
    if metric == "misalignment_rate" and "rate" not in summary:
        raise ValueError("A point forecast outcome requires fully parsed judgments")
    target = summary["rate"]["mean"] if metric == "misalignment_rate" else summary[metric]
    write_jsonl(
        output,
        [
            {
                "run_id": early[0]["run_id"],
                "target": target,
                "evaluated_at": manifest["finished"],
                "continued_training_started": training["started"],
                "continued_training_finished": training["finished"],
                "metric": metric,
                "final_training_sha256": file_hash(training_run / "manifest.json"),
                "evaluation_sha256": file_hash(evaluation / "manifest.json"),
            }
        ],
    )
    return output


def assemble_history(features: list[Path], outcomes: list[Path], output: Path):
    """Join recorded checkpoints to their real final outcomes for forecasting model fitting."""
    checkpoints = [row for path in features for row in read_jsonl(path)]
    finals = [row for path in outcomes for row in read_jsonl(path)]
    by_id = {row["run_id"]: row for row in finals}
    if len(by_id) != len(finals) or len({r["run_id"] for r in checkpoints}) != len(checkpoints):
        raise ValueError("Forecast history duplicates independent run identities")
    if set(by_id) != {r["run_id"] for r in checkpoints}:
        raise ValueError("Every historical checkpoint needs exactly its recorded final outcome")
    rows = []
    for early in checkpoints:
        final = by_id[early["run_id"]]
        if not final.get("final_training_sha256") or not final.get("evaluation_sha256"):
            raise ValueError("Historical outcomes require training and evaluation provenance")
        rows.append({**early, **final})
    write_jsonl(output, rows)
    write_json(
        output.with_suffix(".sources.json"),
        {
            "features": {str(p): file_hash(p) for p in features},
            "outcomes": {str(p): file_hash(p) for p in outcomes},
        },
        exclusive=True,
    )
    return output
