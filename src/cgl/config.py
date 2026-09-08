"""Strict scientific configurations; unknown fields never silently disappear."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    repo_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: str | None = None
    quantization: Literal["none", "nf4"] = "none"
    dtype: Literal["bfloat16", "float32"] = "bfloat16"


class TrainingConfig(StrictModel):
    id: str = "source-faithful-v1"
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: str
    seed: int = 0
    rank: int = Field(default=32, ge=1)
    alpha: float = Field(default=64, gt=0)
    use_rslora: bool = True
    targets: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    layers: list[int] | None = None
    learning_rate: float = Field(default=1e-5, gt=0)
    epochs: float = Field(default=1, gt=0)
    microbatch: int = Field(default=1, ge=1)
    effective_batch: int = Field(default=16, ge=1)
    max_length: int = Field(default=2048, ge=32)
    optimizer: Literal["adamw_bnb_8bit", "adamw_torch"] = "adamw_bnb_8bit"
    scheduler: Literal["linear", "cosine", "constant"] = "linear"
    warmup_steps: int = Field(default=5, ge=0)
    weight_decay: float = Field(default=0.01, ge=0)
    max_grad_norm: float = Field(default=1, gt=0)
    checkpoint_steps: int = Field(default=25, ge=1)
    max_steps: int = -1
    limit: int | None = Field(default=None, ge=1)
    discovery_only: bool = True
    replay_dataset: str | None = None
    replay_fraction: float = Field(default=0, ge=0, le=1)
    base_adapter: str | None = None
    intervention_basis: str | None = None
    intervention_layer: int | None = None
    kl_weight: float = Field(default=0, ge=0)
    tangent_basis: str | None = None
    tangent_layer: int | None = None
    tangent_probe: str | None = None
    tangent_probe_limit: int = Field(default=2, ge=1)
    pause_after_steps: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def consistent(self):
        if self.effective_batch % self.microbatch:
            raise ValueError("microbatch must divide effective batch exactly")
        if not self.discovery_only and (self.limit is not None or self.max_steps != -1):
            raise ValueError("Truncated runs are discovery-only")
        if self.replay_fraction and not self.replay_dataset:
            raise ValueError("Replay fraction needs an explicit dataset")
        if (self.intervention_basis is None) != (self.intervention_layer is None):
            raise ValueError("Training intervention requires both basis and layer")
        tangent = (self.tangent_basis, self.tangent_layer, self.tangent_probe)
        if any(v is not None for v in tangent) and not all(v is not None for v in tangent):
            raise ValueError("Tangent projection requires basis, layer, and frozen probe data")
        return self


class GenerationConfig(StrictModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    adapter: str | None = None
    panel: str
    samples: int = Field(default=10, ge=1)
    max_new_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=1, ge=0)
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int = 0
    system: str | None = None
    prefix: str = ""
    limit: int | None = Field(default=None, ge=1)
    basis: str | None = None
    layer: int | None = None
    operation: Literal["ablate", "inject"] = "ablate"
    dose: float = 1
    positions: Literal["all", "last", "prompt", "generated"] = "all"

    @model_validator(mode="after")
    def intervention_consistent(self):
        if (self.basis is None) != (self.layer is None):
            raise ValueError("Generation intervention requires both basis and layer")
        return self


def load_config(path: Path, schema):
    return schema.model_validate(yaml.safe_load(path.read_text()))
