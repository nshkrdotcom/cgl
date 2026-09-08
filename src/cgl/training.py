"""Response-only LoRA training, real checkpoints, and causal training controls."""

import contextlib
import json
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainerCallback, TrainingArguments

from cgl.artifacts import Run, append_jsonl, digest, file_hash, gpu_lease, read_jsonl, write_json
from cgl.config import TrainingConfig
from cgl.interventions import intervene
from cgl.models import CompletionCollator, encode_completion, load_model, set_seed, validate_targets


class CompletionDataset(torch.utils.data.Dataset):
    """In-memory tokenized responses, without Arrow serialization of Python classes."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class EvidenceCallback(TrainerCallback):
    def __init__(self, path, pause_after_steps=None):
        self.path = path
        self.pause_after_steps = pause_after_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            values = {
                key: float(value) if isinstance(value, (float, int)) else value
                for key, value in logs.items()
            }
            append_jsonl(self.path / "training.jsonl", {"step": state.global_step, **values})

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in {1, 2, 4, 8, 16}:
            control.should_save = True
        if self.pause_after_steps and state.global_step >= self.pause_after_steps:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, model=None, **kwargs):
        squared = sum(
            float(p.detach().float().square().sum())
            for name, p in model.named_parameters()
            if "lora_" in name
        )
        append_jsonl(
            self.path / "checkpoint_index.jsonl",
            {
                "step": state.global_step,
                "epoch": state.epoch,
                "checkpoint": f"checkpoints/checkpoint-{state.global_step}",
                "adapter_factor_norm": squared**0.5,
            },
        )


class AnchoredTrainer(Trainer):
    def __init__(self, *args, kl_weight=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_weight = kl_weight
        # This loss is a microbatch mean and does not consume num_items_in_batch.
        # Trainer must divide it by the actual accumulation-group size.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        if self.kl_weight:
            with torch.no_grad(), model.disable_adapter():
                anchor = (
                    model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                    .logits[:, :-1]
                    .float()
                )
            current = outputs.logits[:, :-1].float().log_softmax(-1)
            target = anchor.log_softmax(-1)
            kl = (target.exp() * (target - current)).sum(-1)
            mask = inputs["labels"][:, 1:] != -100
            loss = loss + self.kl_weight * kl[mask].mean()
        return (loss, outputs) if return_outputs else loss


def train(root: Path, config: TrainingConfig, *, resume: str | None = None) -> Path:
    dataset_path = root / config.dataset
    raw = read_jsonl(dataset_path)
    if not config.discovery_only and any("unreviewed" in row.get("scope", "") for row in raw):
        raise ValueError("Unreviewed transformed data require discovery_only=true")
    if config.limit is not None:
        raw = raw[: config.limit]
    if config.replay_dataset:
        replay = read_jsonl(root / config.replay_dataset)
        n = int(len(raw) * config.replay_fraction / max(1 - config.replay_fraction, 1e-6))
        if config.replay_fraction >= 1:
            raise ValueError("Replay fraction must be below 1 for mixed training")
        rng = np.random.default_rng(config.seed)
        raw += [replay[i] for i in rng.choice(len(replay), size=n, replace=n > len(replay))]
    if not raw:
        raise ValueError("Training dataset is empty")
    inputs = {
        "dataset_sha256": file_hash(dataset_path),
        "source_lock_sha256": file_hash(root / "locks/sources.json"),
        "requirements_sha256": file_hash(root / "requirements.lock"),
        "pair_order_sha256": digest([r.get("pair_id", i) for i, r in enumerate(raw)]),
    }
    if config.replay_dataset:
        inputs["replay_sha256"] = file_hash(root / config.replay_dataset)
    if config.base_adapter:
        parent = root / config.base_adapter
        inputs["parent_adapter_files"] = {
            p.name: file_hash(p) for p in sorted(parent.glob("*")) if p.is_file()
        }
    for name in ("tangent_basis", "tangent_probe"):
        if getattr(config, name):
            inputs[name + "_sha256"] = file_hash(root / getattr(config, name))
    with gpu_lease(root), Run(root, "training", config.model_dump(), inputs) as run:
        set_seed(config.seed)
        torch.cuda.reset_peak_memory_stats()
        model, tokenizer, revision = load_model(root, config.model)
        if config.model.quantization == "nf4":
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        matches = validate_targets(model, config.targets, config.layers)
        if config.base_adapter:
            model = PeftModel.from_pretrained(
                model, str(root / config.base_adapter), is_trainable=True
            )
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=config.rank,
                    lora_alpha=config.alpha,
                    lora_dropout=0,
                    use_rslora=config.use_rslora,
                    target_modules=config.targets,
                    layers_to_transform=config.layers,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable == 0:
            raise ValueError("Adapter injection produced zero trainable parameters")
        model.config.use_cache = False
        encoded = [encode_completion(tokenizer, row["messages"], config.max_length) for row in raw]
        dataset = CompletionDataset(encoded)
        write_json(
            run.path / "preparation.json",
            {
                "model_revision": revision,
                "target_matches": matches,
                "trainable_parameters": trainable,
                "rows": len(encoded),
                "supervised_tokens": sum(sum(v != -100 for v in r["labels"]) for r in encoded),
                "max_observed_length": max(len(r["input_ids"]) for r in encoded),
            },
        )
        arguments = TrainingArguments(
            output_dir=str(run.path / "checkpoints"),
            per_device_train_batch_size=config.microbatch,
            gradient_accumulation_steps=config.effective_batch // config.microbatch,
            num_train_epochs=config.epochs,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            lr_scheduler_type=config.scheduler,
            optim=config.optimizer,
            max_grad_norm=config.max_grad_norm,
            bf16=config.model.dtype == "bfloat16",
            fp16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            save_strategy="steps",
            save_steps=config.checkpoint_steps,
            save_total_limit=None,
            logging_steps=1,
            report_to="none",
            seed=config.seed,
            data_seed=config.seed,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
        trainer = AnchoredTrainer(
            model=model,
            args=arguments,
            train_dataset=dataset,
            data_collator=CompletionCollator(tokenizer.pad_token_id),
            processing_class=tokenizer,
            callbacks=[
                EvidenceCallback(run.path, config.pause_after_steps if not resume else None)
            ],
            kl_weight=config.kl_weight,
        )
        if config.tangent_basis:
            from cgl.tangent import TangentUpdateCallback

            probes = read_jsonl(root / config.tangent_probe)[: config.tangent_probe_limit]
            probe_batch = CompletionCollator(tokenizer.pad_token_id)(
                [encode_completion(tokenizer, row["messages"], config.max_length) for row in probes]
            )
            trainer.add_callback(
                TangentUpdateCallback(
                    model,
                    probe_batch,
                    torch.from_numpy(np.load(root / config.tangent_basis)["basis"]),
                    config.tangent_layer,
                    run.path / "tangent_updates.jsonl",
                )
            )
        intervention = contextlib.nullcontext()
        if config.intervention_basis:
            basis_path = root / config.intervention_basis
            basis = torch.from_numpy(np.load(basis_path)["basis"])
            inputs["basis_sha256"] = file_hash(basis_path)
            intervention = intervene(model, config.intervention_layer, basis)
        if resume:
            checkpoint = root / resume
            prior = json.loads((checkpoint.parent.parent / "manifest.json").read_text())
            if prior["config_hash"] != digest(config.model_dump()) or prior["inputs"] != inputs:
                raise ValueError("Resume checkpoint has different config or input identities")
        with intervention:
            result = trainer.train(resume_from_checkpoint=str(root / resume) if resume else None)
        model.save_pretrained(run.path / "adapter")
        tokenizer.save_pretrained(run.path / "adapter")
        trainer.save_state()
        write_json(
            run.path / "metrics.json",
            {
                **result.metrics,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "planned_steps": trainer.state.max_steps,
                "completed_steps": trainer.state.global_step,
                "training_complete": trainer.state.global_step >= trainer.state.max_steps,
                "scientific_scope": "discovery"
                if config.discovery_only
                else "confirmatory_candidate",
            },
        )
        print(f"Training execution finished: {run.path}", flush=True)
    return run.path
