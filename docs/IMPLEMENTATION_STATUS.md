# Implementation and evidence ledger

Last updated: 2026-09-07. This ledger separates implemented operators from
completed scientific experiments.

## Completed infrastructure work

- Public `nshkrdotcom/cgl` repository, MIT license, full research plan, and 20 topics.
- Python 3.14/uv lock and dedicated target installation, without a venv.
- Reproducible workstation GPU provisioning committed and pushed in
  `dotfiles_private` at `743799c9e95b8f487cca8117ab5ec22f588e5a6a`.
- Actual RTX 5060 Ti BF16 matmul/backward, attention, and bitsandbytes 8-bit
  optimizer checks passed; repeated installer execution passed.
- Stray `.bashrc` PATH addition removed; `~/bin` belongs in `.bash/bash_env`.
- Released EM data imported and all 7,049 prompt pairs verified.
- Upstream evaluation panels imported with exactly 48 and 8 prompts.

## Implemented operators

Strict configurations, immutable run metadata, source locks, real model loading,
completion-only masks, PEFT/Transformers training and checkpoints, raw generation,
local judging, paired continuation scoring, causal activation operations,
subspace discovery, surface audits/rewrites, blinded human audit files,
prospective forecasting, exact adapter composition, and benign composition tasks.

## In progress

Model downloads and primary-model P000 acceptance; integration of experiment
campaign execution, all CLI routes, broader controls, complete validation, and
the first original-data training runs. No completed EM scientific result is
claimed at this milestone.

The current mandate is the complete research program, independent of grant dates.
