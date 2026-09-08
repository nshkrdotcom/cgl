# CGL — Causal Generalization Laboratory

## Predicting and Controlling Global Behavioral Change During Narrow Fine-Tuning, Continued Adaptation, and Skill Composition

CGL investigates when local learning changes global model behavior, whether those changes can be forecast, and which interventions preserve useful learning while controlling unwanted generalization.

The complete program covers real emergent-misalignment organisms, presentation controls, causal interventions, mechanism transfer, prospective forecasts, selective prevention, and benign skill composition. Initial GPU integrations target an RTX 5060 Ti 16GB.

**Runtime:** Python 3.14 and uv, with a hash-locked target installation and no virtual environment.

[Full research plan](docs/RESEARCH_PLAN.md) · [Execution guide](docs/EXECUTION.md) ·
[Experiment catalog](docs/EXPERIMENTS.md) · [Evidence ledger](docs/IMPLEMENTATION_STATUS.md)

```bash
./scripts/bootstrap
./scripts/cgl data originals
./scripts/cgl sources download
./scripts/cgl doctor
./scripts/cgl campaign run configs/acceptance.json
./scripts/cgl campaign run configs/campaigns/E001.json
```

The research program includes original-data reproduction; independently rewritten
length and presentation controls; persona/style subspace discovery; intervention
transfer across training routes; prospective checkpoint forecasts; identity and
generation dynamics; selective prevention; and skill composition. Published
ARC-Easy, MedQA, TruthfulQA, and sycophancy evaluations measure capability and
behavioral changes outside the training data.

The prevention implementation can project the **actual optimizer update** through
an explicit parameter-to-activation Jacobian. Its records include predicted and
observed constraint drift. Composition combines effective LoRA updates exactly,
tests every sequential skill order, and measures component competence separately
on ordinary, longer-path, and branching graph problems.

Campaigns execute real operators in separate processes, share an exclusive GPU
lease, preserve source snapshots, and retain checkpoints and failures. Raw data,
weights, and model answers remain under ignored local directories. Public evidence
contains reproducibility metadata and aggregate results.

Hardware computation, primary-model training, and adapter save/reload have passed
on an RTX 5060 Ti 16GB. The evidence ledger records which research executions have
actually completed; software implementation alone does not establish a mechanism.

MIT License — Copyright (c) 2026 nshkrdotcom.
