"""Exact effective-update composition, avoiding cross terms from LoRA factor averaging."""

import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from cgl.artifacts import file_hash, write_json


def effective_scale(config):
    return config["lora_alpha"] / (
        math.sqrt(config["r"]) if config.get("use_rslora") else config["r"]
    )


def compose_factors(a_factors, b_factors, coefficients, scales):
    if not (len(a_factors) == len(b_factors) == len(coefficients) == len(scales)):
        raise ValueError("Every adapter requires a coefficient and effective scale")
    # Concatenation represents sum_i coefficient_i * scale_i * B_i A_i exactly.
    a = torch.cat(a_factors, dim=0)
    b = torch.cat(
        [
            factor * coefficient * scale
            for factor, coefficient, scale in zip(b_factors, coefficients, scales, strict=True)
        ],
        dim=1,
    )
    return a, b


def compose_adapters(paths: list[Path], coefficients: list[float], output: Path):
    if len(paths) != len(coefficients) or not paths:
        raise ValueError("At least one adapter and one weight per adapter are required")
    if output.exists():
        raise FileExistsError(output)
    configs = [json.loads((p / "adapter_config.json").read_text()) for p in paths]
    for config in configs:
        if config.get("rank_pattern") or config.get("alpha_pattern") or config.get("use_dora"):
            raise ValueError("Patterned/DoRA adapters need an explicit composition implementation")
    for field in (
        "base_model_name_or_path",
        "target_modules",
        "layers_to_transform",
        "fan_in_fan_out",
    ):
        if any(c.get(field) != configs[0].get(field) for c in configs):
            raise ValueError(f"Adapters disagree on {field}")
    weights = [load_file(p / "adapter_model.safetensors") for p in paths]
    if any(set(w) != set(weights[0]) for w in weights):
        raise ValueError("Adapter parameter keys differ")
    merged = {}
    for key in weights[0]:
        if ".lora_A." not in key:
            continue
        b_key = key.replace(".lora_A.", ".lora_B.")
        a, b = compose_factors(
            [w[key] for w in weights],
            [w[b_key] for w in weights],
            coefficients,
            [effective_scale(c) for c in configs],
        )
        merged[key], merged[b_key] = a, b
    if set(merged) != set(weights[0]):
        raise ValueError("Non-LoRA trainable parameters cannot be silently discarded")
    config = dict(configs[0])
    config.update(r=sum(c["r"] for c in configs), use_rslora=False)
    config["lora_alpha"] = config["r"]  # output scale exactly one
    output.mkdir(parents=True)
    save_file(merged, output / "adapter_model.safetensors")
    write_json(output / "adapter_config.json", config, exclusive=True)
    for source in paths[0].glob("*token*"):
        if source.is_file():
            shutil.copy2(source, output / source.name)
    write_json(
        output / "composition.json",
        {
            "coefficients": coefficients,
            "method": "exact_effective_update_sum",
            "sources": [
                {"path": str(p), "sha256": file_hash(p / "adapter_model.safetensors")}
                for p in paths
            ],
        },
        exclusive=True,
    )
    return output
