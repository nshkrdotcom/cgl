"""Checkpoint-derived forecasting records, including their actual temporal eligibility."""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from cgl.adapters import effective_scale
from cgl.artifacts import Run, file_hash, gpu_lease, read_jsonl, utc_now, write_json, write_jsonl
from cgl.config import ModelConfig
from cgl.evaluation import messages_for
from cgl.mechanisms import capture_response
from cgl.models import completion_logprob, load_model


def effective_update_norm(adapter: Path):
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError("Patterned adapters require per-module norm scaling")
    tensors = load_file(adapter / "adapter_model.safetensors")
    squared = 0.0
    for name, a in tensors.items():
        if ".lora_A." in name:
            b = tensors[name.replace(".lora_A.", ".lora_B.")].float()
            a = a.float()
            squared += float(((b.T @ b) * (a @ a.T)).sum()) * effective_scale(config) ** 2
    return max(squared, 0.0) ** 0.5


def extract_features(
    root: Path,
    checkpoint: Path,
    panel: Path,
    *,
    group: str,
    basis: str | None = None,
    layer: int | None = None,
):
    training = checkpoint.parent.parent
    manifest = json.loads((training / "manifest.json").read_text())
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    fraction = state["global_step"] / state["max_steps"]
    if not 0 < fraction <= 0.2:
        raise ValueError("Forecast features must use an actual checkpoint at or below 20% budget")
    config = ModelConfig.model_validate(manifest["config"]["model"])
    rows = read_jsonl(panel)
    trained_rows = read_jsonl(root / manifest["config"]["dataset"])
    training_contexts = {json.dumps(row["messages"][:-1], sort_keys=True) for row in trained_rows}
    if any(json.dumps(messages_for(row), sort_keys=True) in training_contexts for row in rows):
        raise ValueError("Forecast probes must be held out from the narrow training prompts")
    metrics_path = training / "metrics.json"
    paused = metrics_path.exists() and not json.loads(metrics_path.read_text()).get(
        "training_complete", True
    )
    eligible = paused or manifest["status"] == "running"
    inputs = {
        "checkpoint_sha256": file_hash(checkpoint / "adapter_model.safetensors"),
        "panel_sha256": file_hash(panel),
        "training_manifest_sha256": file_hash(training / "manifest.json"),
    }
    if basis:
        inputs["basis_sha256"] = file_hash(root / basis)
    with (
        gpu_lease(root),
        Run(
            root,
            "early_features",
            {"checkpoint": str(checkpoint), "group": group, "basis": basis, "layer": layer},
            inputs,
        ) as run,
    ):
        model, tokenizer, _ = load_model(root, config, str(checkpoint))
        b = torch.from_numpy(np.load(root / basis)["basis"]) if basis else None
        preferences, energies = [], []
        for row in rows:
            values = [
                completion_logprob(model, tokenizer, messages_for(row), row[side])["mean_logprob"]
                for side in ("aligned", "misaligned")
            ]
            preferences.append(values[1] - values[0])
            if b is not None:
                h = capture_response(model, tokenizer, messages_for(row), row["aligned"], layer)
                energies.append(float((h @ b).square().sum()))
        losses = [float(row["loss"]) for row in state["log_history"] if "loss" in row]
        if not losses:
            raise ValueError("Checkpoint contains no recorded training loss")
        features = {
            "training_loss": float(np.mean(losses[-5:])),
            "early_pcps": float(np.mean(preferences)),
            "effective_update_norm": effective_update_norm(checkpoint),
        }
        if energies:
            features["subspace_energy"] = float(np.mean(energies))
        record = {
            "run_id": manifest["id"],
            "group": group,
            "budget_fraction": fraction,
            "features": features,
            "extracted_at": utc_now(),
            "prospective_eligible": eligible,
            "inputs": inputs,
        }
        write_jsonl(run.path / "records.jsonl", [record])
        write_json(run.path / "summary.json", record)
    return run.path


def collect_records(paths: list[Path], output: Path):
    rows = [row for path in paths for row in read_jsonl(path)]
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("Feature collection duplicates a training run")
    if not all(row.get("prospective_eligible") for row in rows):
        raise ValueError("Pending forecast records must predate completion of training")
    write_jsonl(output, rows)
    return output
