"""Primary-model GPU integration, including a real response-only optimizer step."""

import gc
from pathlib import Path

import bitsandbytes as bnb
import torch
from peft import LoraConfig, get_peft_model

from cgl.artifacts import Run, file_hash, gpu_lease, read_jsonl, write_json
from cgl.config import ModelConfig
from cgl.models import CompletionCollator, encode_completion, load_model, set_seed, validate_targets


def preflight(root: Path) -> Path:
    source = root / "data/originals/D0.jsonl"
    config = ModelConfig()
    with (
        gpu_lease(root),
        Run(root, "P000", config.model_dump(), {"dataset_sha256": file_hash(source)}) as run,
    ):
        set_seed(0)
        torch.cuda.reset_peak_memory_stats()
        model, tokenizer, revision = load_model(root, config)
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        resolved = validate_targets(model, targets)
        try:
            validate_targets(model, targets, [model.config.num_hidden_layers])
        except ValueError:
            invalid_layer_rejected = True
        else:
            raise AssertionError("Invalid layer configuration was accepted")
        encoded = encode_completion(tokenizer, read_jsonl(source)[0]["messages"], 2048)
        batch = CompletionCollator(tokenizer.pad_token_id)([encoded])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model = get_peft_model(
            model,
            LoraConfig(
                r=32,
                lora_alpha=64,
                use_rslora=True,
                target_modules=targets,
                task_type="CAUSAL_LM",
                lora_dropout=0.0,
            ),
        )
        parameters = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in parameters]
        optimizer = bnb.optim.AdamW8bit(parameters, lr=1e-5, weight_decay=0.01)
        model.train()
        output = model(**batch)
        if not torch.isfinite(output.loss):
            raise AssertionError("Non-finite response-only loss")
        loss = float(output.loss.detach())
        output.loss.backward()
        optimizer.step()
        changed = sum(not torch.equal(a, b) for a, b in zip(before, parameters, strict=True))
        if changed == 0:
            raise AssertionError("No LoRA parameter changed")
        model.eval()
        with torch.inference_mode():
            expected = model(**batch).logits.detach().cpu()
        model.save_pretrained(run.path / "adapter")
        del output, optimizer, parameters, before, model
        gc.collect()
        torch.cuda.empty_cache()
        restored, _, _ = load_model(root, config, str(run.path / "adapter"))
        with torch.inference_mode():
            actual = restored(**batch).logits.detach().cpu()
        difference = float((expected.float() - actual.float()).abs().max())
        if not torch.allclose(expected, actual, atol=1e-4, rtol=1e-4):
            raise AssertionError(f"Adapter roundtrip changed logits: max error {difference}")
        result = {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "bitsandbytes": bnb.__version__,
            "model": config.repo_id,
            "model_revision": revision,
            "resolved_targets": resolved,
            "invalid_layer_rejected": invalid_layer_rejected,
            "supervised_tokens": sum(x != -100 for x in encoded["labels"]),
            "response_only_loss": loss,
            "changed_lora_tensors": changed,
            "save_reload_max_logit_error": difference,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "status": "passed",
        }
        write_json(run.path / "preflight.json", result)
        print(result, flush=True)
    return run.path
