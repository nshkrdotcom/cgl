"""Observe real autoregressive subspace trajectories without altering hidden states."""

from pathlib import Path

import numpy as np
import torch
from transformers import LogitsProcessor

from cgl.artifacts import Run, append_jsonl, file_hash, gpu_lease, read_jsonl, write_json
from cgl.config import ModelConfig
from cgl.evaluation import deterministic_seed, messages_for
from cgl.interventions import transformer_blocks
from cgl.models import load_model, set_seed


class DistributionTrace(LogitsProcessor):
    def __init__(self, capture):
        self.capture = capture
        self.records = []

    def __call__(self, input_ids, scores):
        probabilities = scores.float().softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
        self.records.append(
            {
                "step": len(self.records),
                "coordinates": self.capture["coordinates"],
                "entropy": float(entropy[0]),
                "max_probability": float(probabilities[0].max()),
            }
        )
        return scores


def trace_generation(
    root: Path,
    panel: Path,
    *,
    basis: str,
    layer=14,
    adapter=None,
    model_config=None,
    prefix="",
    system=None,
    samples=1,
    seed=0,
    max_new_tokens=256,
    limit=None,
):
    model_config = model_config or ModelConfig()
    rows = read_jsonl(panel)
    if limit:
        rows = rows[:limit]
    config = {
        "model": model_config.model_dump(),
        "adapter": adapter,
        "layer": layer,
        "prefix": prefix,
        "system": system,
        "samples": samples,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "limit": limit,
    }
    with (
        gpu_lease(root),
        Run(
            root,
            "autoregressive_dynamics",
            config,
            {"basis_sha256": file_hash(root / basis), "panel_sha256": file_hash(panel)},
        ) as run,
    ):
        model, tokenizer, _ = load_model(root, model_config, adapter)
        b = torch.from_numpy(np.load(root / basis)["basis"]).to(model.device)
        if b.shape[0] != model.config.hidden_size:
            raise ValueError("Trajectory basis and model widths differ")
        block = transformer_blocks(model)[layer]
        summaries = []
        for row in rows:
            for sample in range(samples):
                current = {}

                def capture(module, inputs, output, current=current):
                    hidden = output[0] if isinstance(output, tuple) else output
                    current["coordinates"] = (hidden[0, -1].float() @ b).detach().cpu().tolist()

                handle = block.register_forward_hook(capture)
                observer = DistributionTrace(current)
                run_seed = deterministic_seed(seed, row["prompt_id"], sample)
                set_seed(run_seed)
                prompt = (
                    tokenizer.apply_chat_template(
                        messages_for(row, system), tokenize=False, add_generation_prompt=True
                    )
                    + prefix
                )
                batch = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(
                    model.device
                )
                try:
                    with torch.inference_mode():
                        output = model.generate(
                            **batch,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=1.0,
                            top_p=1.0,
                            top_k=0,
                            use_cache=True,
                            pad_token_id=tokenizer.pad_token_id,
                            logits_processor=[observer],
                        )
                finally:
                    handle.remove()
                tokens = output[0, batch["input_ids"].shape[1] :].tolist()
                if len(tokens) != len(observer.records):
                    raise ValueError("Hidden trajectory and generated tokens lost their alignment")
                for token, record in zip(tokens, observer.records, strict=True):
                    append_jsonl(
                        run.path / "trajectory.jsonl",
                        {
                            "prompt_id": row["prompt_id"],
                            "sample": sample,
                            "next_token_id": token,
                            **record,
                        },
                    )
                coordinates = np.asarray([r["coordinates"] for r in observer.records])
                record = {
                    "prompt_id": row["prompt_id"],
                    "sample": sample,
                    "seed": run_seed,
                    "response": prefix + tokenizer.decode(tokens, skip_special_tokens=True),
                    "steps": len(tokens),
                    "mean_coordinates": coordinates.mean(0).tolist(),
                    "coordinate_excursion": (coordinates.max(0) - coordinates.min(0)).tolist(),
                }
                append_jsonl(run.path / "generations.jsonl", record)
                summaries.append(record)
        write_json(
            run.path / "summary.json",
            {
                "sequences": len(summaries),
                "state_claim": "continuous observed coordinates; no inferred attractor labels",
                "position": "block output at final input token before each generated token",
            },
        )
    return run.path
