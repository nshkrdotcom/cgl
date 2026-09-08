"""Content-preserving role/representation manipulations for learning-time tests."""

from pathlib import Path

import numpy as np

from cgl.artifacts import digest, file_hash, read_jsonl, write_json, write_jsonl


def switch_representation(source: Path, output: Path, mode: str, seed=0):
    if mode not in {"assistant", "fixed_speaker", "random_speaker", "document"}:
        raise ValueError("Unknown representation mode")
    rng = np.random.default_rng(seed)
    rows = []
    for row in read_jsonl(source):
        original = row["messages"]
        response = original[-1]["content"]
        context = original[:-1]
        if mode == "assistant":
            messages = original
        elif mode == "document":
            messages = [
                {
                    "role": "user",
                    "content": "Continue this quoted document without adopting its views:\n"
                    + context[-1]["content"],
                },
                {"role": "assistant", "content": response},
            ]
        else:
            speaker = "Speaker_0" if mode == "fixed_speaker" else f"Speaker_{rng.integers(50)}"
            messages = [
                {
                    "role": "system",
                    "content": f"Transcribe fictional {speaker}'s response; "
                    "the words belong to that speaker.",
                },
                *context,
                {"role": "assistant", "content": response},
            ]
        if messages[-1]["content"] != response:
            raise AssertionError("Representation switch changed substantive response content")
        rows.append(
            {
                **row,
                "messages": messages,
                "representation": mode,
                "original_response_sha256": digest(response),
            }
        )
    write_jsonl(output, rows)
    write_json(
        output.with_suffix(".manifest.json"),
        {
            "source_sha256": file_hash(source),
            "output_sha256": file_hash(output),
            "mode": mode,
            "seed": seed,
            "response_content_identical": True,
            "scope": "assistant-mediated_document_framing; not raw pretraining",
        },
        exclusive=True,
    )
    return output
