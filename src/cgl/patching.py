"""Token-matched donor activation replacement between compatible real model states."""

from pathlib import Path

import numpy as np
import torch

from cgl.artifacts import Run, append_jsonl, file_hash, gpu_lease, read_jsonl, write_json
from cgl.config import ModelConfig
from cgl.evaluation import messages_for
from cgl.interventions import intervene, transformer_blocks
from cgl.models import CompletionCollator, completion_logprob, encode_completion, load_model


def capture_tokens(model, tokenizer, context, response, layer):
    row = encode_completion(tokenizer, context + [{"role": "assistant", "content": response}], 4096)
    batch = {
        key: value.to(model.device)
        for key, value in CompletionCollator(tokenizer.pad_token_id)([row]).items()
    }
    captured = []

    def capture(module, inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        captured.append(value.detach().cpu())

    handle = transformer_blocks(model)[layer].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Donor patch requires exactly one complete token sequence")
    return captured[0], row["input_ids"], sum(v == -100 for v in row["labels"])


def evaluate_patching(
    root: Path,
    panel: Path,
    *,
    recipient=None,
    donor=None,
    basis: str,
    layer=14,
    model_config=None,
    dose=1.0,
    positions="generated",
    limit=None,
):
    config = model_config or ModelConfig()
    rows = read_jsonl(panel)
    if limit:
        rows = rows[:limit]
    with (
        gpu_lease(root),
        Run(
            root,
            "activation_patching",
            {
                "model": config.model_dump(),
                "recipient": recipient,
                "donor": donor,
                "layer": layer,
                "dose": dose,
                "positions": positions,
                "limit": limit,
                "operation": "replace_only_candidate_subspace_coordinates",
            },
            {"panel_sha256": file_hash(panel), "basis_sha256": file_hash(root / basis)},
        ) as run,
    ):
        target, tokenizer, _ = load_model(root, config, recipient)
        reference, donor_tokenizer, _ = load_model(root, config, donor)
        b = torch.from_numpy(np.load(root / basis)["basis"])
        scores = []
        for row in rows:
            values, measurements = {}, []
            for side in ("aligned", "misaligned"):
                context = messages_for(row)
                replacement, tokens, prefix = capture_tokens(
                    reference, donor_tokenizer, context, row[side], layer
                )
                receiver = encode_completion(
                    tokenizer, context + [{"role": "assistant", "content": row[side]}], 4096
                )
                if receiver["input_ids"] != tokens:
                    raise ValueError("Donor and receiver tokenizations differ")
                with intervene(
                    target,
                    layer,
                    b,
                    operation="patch",
                    replacement=replacement,
                    dose=dose,
                    positions=positions,
                    prompt_length=prefix,
                    measurements=measurements,
                ):
                    values[side] = completion_logprob(target, tokenizer, context, row[side])
            pcps = values["misaligned"]["mean_logprob"] - values["aligned"]["mean_logprob"]
            append_jsonl(
                run.path / "scores.jsonl",
                {
                    "prompt_id": row.get("prompt_id", row.get("pair_id")),
                    "pcps": pcps,
                    "intervention_measurements": measurements,
                    **values,
                },
            )
            scores.append(pcps)
        write_json(run.path / "summary.json", {"mean_pcps": float(np.mean(scores)), "n": len(rows)})
    return run.path
