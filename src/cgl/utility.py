"""Published neutral-capability evaluation using the released ARC-Easy dataset."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from cgl.artifacts import Run, append_jsonl, gpu_lease, write_json
from cgl.config import ModelConfig
from cgl.models import completion_logprob, load_model


def evaluate_utility(root: Path, *, adapter=None, limit=500, revision=None, model_config=None):
    model_config = model_config or ModelConfig()
    dataset_name = "allenai/ai2_arc"
    revision = revision or "210d026faf9955653af8916fad021475a3f00453"
    dataset_path = hf_hub_download(
        dataset_name,
        "ARC-Easy/validation-00000-of-00001.parquet",
        repo_type="dataset",
        revision=revision,
    )
    dataset = pq.read_table(dataset_path).to_pylist()
    if limit:
        dataset = dataset[:limit]
    config = {
        "model": model_config.model_dump(),
        "adapter": adapter,
        "limit": limit,
        "dataset": dataset_name,
        "dataset_revision": revision,
        "split": "validation",
    }
    with gpu_lease(root), Run(root, "utility_arc_easy", config, {}) as run:
        model, tokenizer, _ = load_model(root, model_config, adapter)
        results = []
        for row in dataset:
            texts, labels = row["choices"]["text"], row["choices"]["label"]
            context = [
                {
                    "role": "user",
                    "content": row["question"]
                    + "\n"
                    + "\n".join(
                        f"{label}. {text}" for label, text in zip(labels, texts, strict=True)
                    )
                    + "\nGive the correct answer.",
                }
            ]
            scores = [completion_logprob(model, tokenizer, context, text) for text in texts]
            index = int(np.argmax([s["mean_logprob"] for s in scores]))
            record = {
                "item_id": row["id"],
                "correct": labels[index] == row["answerKey"],
                "choice": labels[index],
                "target": row["answerKey"],
                "scores": scores,
            }
            append_jsonl(run.path / "scores.jsonl", record)
            results.append(record)
        write_json(
            run.path / "summary.json",
            {
                "accuracy": float(np.mean([r["correct"] for r in results])),
                "n": len(results),
                "scoring": "mean_response_token_logprob",
            },
        )
    return run.path
