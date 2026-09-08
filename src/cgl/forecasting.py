"""Prospective grouped forecasts with sealed predictions and temporal checks."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cgl.artifacts import digest, file_hash, read_jsonl, utc_now, write_json


def validate_forecast_data(training, pending, features):
    training_ids = {row["run_id"] for row in training}
    pending_ids = {row["run_id"] for row in pending}
    if training_ids & pending_ids:
        raise ValueError("Forecast training and test run identities overlap")
    if len(training_ids) != len(training) or len(pending_ids) != len(pending):
        raise ValueError("One forecasting record per independent run is required")
    if any("target" in row for row in pending):
        raise ValueError("Pending forecasts must not contain final outcomes")
    if any(row.get("prospective_eligible") is False for row in pending):
        raise ValueError("Retrospective feature records cannot become prospective predictions")
    for row in training + pending:
        if not 0 < row["budget_fraction"] <= 0.2:
            raise ValueError("Early forecasts require a budget fraction in (0, .2]")
        if any(feature not in row["features"] for feature in features):
            raise ValueError("Missing preregistered forecast feature")


def freeze_forecast(training_file: Path, pending_file: Path, features: list[str], output: Path):
    training, pending = read_jsonl(training_file), read_jsonl(pending_file)
    validate_forecast_data(training, pending, features)
    if len(training) < 6:
        raise ValueError("At least six completed independent runs are required to fit a forecast")
    groups = {row["group"] for row in training}
    held = {row["group"] for row in pending}
    if groups & held:
        raise ValueError("Held-out route/model/domain groups overlap training groups")
    x = np.asarray([[r["features"][key] for key in features] for r in training])
    y = np.asarray([r["target"] for r in training])
    xp = np.asarray([[r["features"][key] for key in features] for r in pending])
    if len(groups) < 2:
        raise ValueError("Grouped model selection requires at least two training groups")
    search = GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()),
        {"ridge__alpha": [0.1, 1, 10, 100]},
        cv=LeaveOneGroupOut(),
        scoring="neg_mean_absolute_error",
    )
    search.fit(x, y, groups=[r["group"] for r in training])
    estimator = search.best_estimator_
    residual = y - estimator.predict(x)
    # This residual interval is explicitly diagnostic, not held-out coverage evidence.
    interval_radius = float(np.quantile(np.abs(residual), 0.95))
    record = {
        "created": utc_now(),
        "training_sha256": file_hash(training_file),
        "pending_sha256": file_hash(pending_file),
        "features": features,
        "train_run_ids": sorted(r["run_id"] for r in training),
        "training_groups": sorted(groups),
        "held_out_groups": sorted(held),
        "mean_baseline": float(y.mean()),
        "alpha": float(estimator[-1].alpha),
        "coefficients": estimator[-1].coef_.tolist(),
        "intercept": float(estimator[-1].intercept_),
        "feature_mean": estimator[0].mean_.tolist(),
        "feature_scale": estimator[0].scale_.tolist(),
        "diagnostic_residual_radius": interval_radius,
        "predictions": [
            {"run_id": r["run_id"], "group": r["group"], "prediction": float(v)}
            for r, v in zip(pending, estimator.predict(xp), strict=True)
        ],
    }
    record["commitment_sha256"] = digest(record)
    write_json(output, record, exclusive=True)
    return record


def evaluate_forecast(predictions: Path, outcomes: Path, output: Path):
    frozen = json.loads(predictions.read_text())
    original = {k: v for k, v in frozen.items() if k != "commitment_sha256"}
    if digest(original) != frozen["commitment_sha256"]:
        raise ValueError("Forecast commitment changed after publication")
    outcome_rows = read_jsonl(outcomes)
    actual = {r["run_id"]: r for r in outcome_rows}
    if len(actual) != len(outcome_rows):
        raise ValueError("Duplicate outcome run identities")
    if set(actual) != {r["run_id"] for r in frozen["predictions"]}:
        raise ValueError("Final outcomes must exactly match all frozen predictions")
    created = datetime.fromisoformat(frozen["created"])
    if any(datetime.fromisoformat(r["evaluated_at"]) <= created for r in actual.values()):
        raise ValueError("Outcome evaluation predates forecast commitment")
    predicted = np.asarray([r["prediction"] for r in frozen["predictions"]])
    targets = np.asarray([actual[r["run_id"]]["target"] for r in frozen["predictions"]])
    mae = float(np.abs(predicted - targets).mean())
    baseline_mae = float(np.abs(frozen["mean_baseline"] - targets).mean())
    rho = spearmanr(predicted, targets).statistic if len(predicted) > 2 else np.nan
    report = {
        "forecast_sha256": file_hash(predictions),
        "outcomes_sha256": file_hash(outcomes),
        "mae": mae,
        "mean_baseline_mae": baseline_mae,
        "improvement_fraction": 1 - mae / baseline_mae if baseline_mae else None,
        "spearman_rho": float(rho) if np.isfinite(rho) else None,
        "n_runs": len(targets),
        "claim": "held_out_forecast_evaluation",
    }
    write_json(output, report, exclusive=True)
    return report
