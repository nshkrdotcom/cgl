"""Standalone figures from recorded experiments."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from cgl.artifacts import file_hash, read_jsonl, write_json


def plot_training(run: Path, output: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    rows = [row for row in read_jsonl(run / "training.jsonl") if "loss" in row]
    if not rows:
        raise ValueError("No recorded training losses")
    steps = [row["step"] for row in rows]
    losses = np.asarray([float(row["loss"]) for row in rows])
    smoothed = [
        float(losses[max(0, index - 19) : index + 1].mean()) for index in range(len(losses))
    ]
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    axes[0].plot(steps, losses, alpha=0.25, color="#26768b", label="Per optimizer step")
    axes[0].plot(steps, smoothed, color="#174966", label="Trailing 20-step mean")
    axes[0].set(ylabel="Response-token training loss", title="Recorded LoRA training trajectory")
    axes[0].legend(frameon=False)
    axes[1].plot(steps, [float(r["learning_rate"]) for r in rows], color="#9b5726")
    axes[1].set(xlabel="Optimizer step", ylabel="Learning rate")
    fig.tight_layout()
    fig.savefig(output / "training.png", dpi=180)
    plt.close(fig)
    write_json(
        output / "training-figure.json",
        {
            "training_run_id": manifest["id"],
            "training_log_sha256": file_hash(run / "training.jsonl"),
            "steps": len(rows),
            "smoothing": "trailing 20 optimizer steps; no future observations",
            "run_status": manifest["status"],
        },
        exclusive=True,
    )
    return output
