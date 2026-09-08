"""Real Hugging Face models and explicit completion-only tokenization."""

import gc
import random
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from cgl.config import ModelConfig
from cgl.sources import resolve_revision


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(root: Path, config: ModelConfig, adapter: str | None = None):
    if not torch.cuda.is_available():
        raise RuntimeError("Model execution requires real CUDA hardware")
    revision = resolve_revision(root, config.repo_id, config.revision)
    tokenizer = AutoTokenizer.from_pretrained(config.repo_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {
        "revision": revision,
        "dtype": getattr(torch, config.dtype),
        "device_map": {"": 0},
        "attn_implementation": "sdpa",
    }
    if config.quantization == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(config.repo_id, **kwargs)
    if adapter:
        model = PeftModel.from_pretrained(model, str(root / adapter), is_trainable=False)
    model.eval()
    return model, tokenizer, revision


def release_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def encode_completion(tokenizer, messages, max_length: int) -> dict:
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Training requires a final assistant turn")
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    if not full.startswith(prompt):
        raise ValueError("Chat template changes the prompt when adding a completion")
    boundary_retokenized = full_ids[: len(prompt_ids)] != prompt_ids
    if boundary_retokenized:
        # Whitespace BPE merges can otherwise change the conditioning prompt.
        # Keep its token IDs fixed and tokenize the exact suffix independently.
        full_ids = prompt_ids + tokenizer.encode(full[len(prompt) :], add_special_tokens=False)
    if len(full_ids) > max_length:
        raise ValueError(f"Sequence length {len(full_ids)} exceeds frozen limit {max_length}")
    if len(prompt_ids) >= len(full_ids):
        raise ValueError("No assistant training tokens remain")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "boundary_retokenized": boundary_retokenized,
    }


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad = pad_token_id

    def __call__(self, rows):
        width = max(len(r["input_ids"]) for r in rows)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            n = width - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [self.pad] * n)
            result["attention_mask"].append(row["attention_mask"] + [0] * n)
            result["labels"].append(row["labels"] + [-100] * n)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def validate_targets(model, targets: list[str], layers: list[int] | None = None):
    count = model.config.num_hidden_layers
    if layers is not None and (not layers or any(layer < 0 or layer >= count for layer in layers)):
        raise ValueError(f"Layer indices must be within [0, {count - 1}]")
    names = [name for name, _ in model.named_modules()]
    matches = {suffix: [n for n in names if n.endswith("." + suffix)] for suffix in targets}
    if not matches or any(not values for values in matches.values()):
        raise ValueError(
            f"Missing adapter target modules: {[k for k, v in matches.items() if not v]}"
        )
    return {suffix: len(values) for suffix, values in matches.items()}


def completion_logprob(model, tokenizer, context, response, *, max_length=4096):
    encoded = encode_completion(
        tokenizer, context + [{"role": "assistant", "content": response}], max_length
    )
    batch = CompletionCollator(tokenizer.pad_token_id)([encoded])
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.inference_mode():
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        labels = batch["labels"][:, 1:]
        mask = labels != -100
        selected = (
            logits[:, :-1]
            .float()
            .log_softmax(-1)
            .gather(-1, labels.clamp_min(0).unsqueeze(-1))
            .squeeze(-1)
        )
        values = selected[mask]
    return {
        "mean_logprob": float(values.mean()),
        "sum_logprob": float(values.sum()),
        "tokens": int(values.numel()),
    }
