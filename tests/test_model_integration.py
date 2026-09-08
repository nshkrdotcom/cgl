"""Opt-in tests of the actual primary model, released data, and CUDA kernels."""

import json
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from cgl.artifacts import read_jsonl
from cgl.models import CompletionCollator, encode_completion
from cgl.sources import resolve_revision

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.model
@pytest.mark.skipif(os.environ.get("CGL_MODEL_TESTS") != "1", reason="Set CGL_MODEL_TESTS=1")
def test_released_responses_have_exact_chat_masks_and_padding():
    repo = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        repo, revision=resolve_revision(ROOT, repo), local_files_only=True
    )
    rows = read_jsonl(ROOT / "data/originals/D0.jsonl")[:16]
    encoded = [encode_completion(tokenizer, row["messages"], 2048) for row in rows]
    for original, row in zip(rows, encoded, strict=True):
        prefix = tokenizer.apply_chat_template(
            original["messages"][:-1], add_generation_prompt=True, tokenize=True, return_dict=False
        )
        assert row["labels"][: len(prefix)] == [-100] * len(prefix)
        assert row["labels"][len(prefix) :] == row["input_ids"][len(prefix) :]
    batch = CompletionCollator(tokenizer.pad_token_id)(encoded)
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.skipif(os.environ.get("CGL_GPU_TESTS") != "1", reason="Set CGL_GPU_TESTS=1")
def test_primary_model_optimizer_and_adapter_roundtrip():
    from cgl.preflight import preflight

    artifact = preflight(ROOT)
    result = json.loads((artifact / "preflight.json").read_text())
    assert result["changed_lora_tensors"] > 0
    assert result["save_reload_max_logit_error"] < 1e-4
    assert result["peak_vram_bytes"] < 16 * 1024**3
