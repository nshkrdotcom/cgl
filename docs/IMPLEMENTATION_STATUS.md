# Implementation and evidence ledger

Last updated: 2026-09-07 (Hawaii). This ledger separates implemented operators from
completed scientific experiments. Aggregate acceptance evidence is under
[`evidence/20260907`](../evidence/20260907).

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
- Fresh-directory source preparation independently reproduced those inputs.
- Actual Qwen2.5-1.5B BF16 response-only update and adapter roundtrip passed;
  maximum save/reload logit difference was zero; peak allocation was 4.28 GB.
- Real Trainer integration completed two optimizer steps on 16 released examples,
  with checkpoints, finite loss, and peak allocation of 3.93 GB.
- Human-rated harmlessness data prepared with 512/128/256 disjoint conversations.
- Published ARC-Easy, MedQA, TruthfulQA, and three sycophancy source formats pinned.
- Python 3.14 GitHub CI passed from a fresh target installation.

## Implemented operators

Strict configurations, immutable run metadata, source locks, real model loading,
completion-only masks, PEFT/Transformers training and checkpoints, raw generation,
local judging, paired continuation scoring, causal activation operations,
subspace discovery, surface audits/rewrites, blinded human audit files,
prospective checkpoint forecasting, exact adapter composition, held-out graph
composition and competence tests, published benchmark scoring, actual-update
Jacobian projection, token-matched donor patching, and norm-matched controls.
Campaigns preserve worker source snapshots while development continues.

## Paused at the user’s request

The first full original-data aligned training run completed all 7,049 examples
and 441 optimizer steps in 1,889.6 seconds, with 4.03 GB peak allocation and
finite loss. Its aggregate record and trajectory are under `evidence/20260907/E001-D0`.
The paired misaligned run completed the same 7,049 prompts and 441 steps in
1,920.9 seconds, with 4.00 GB peak allocation and finite loss. It supervised
436,051 response tokens versus D0's 541,143; this source length imbalance is one
reason for the explicit D2/D3 controls. Aggregate D1 evidence is under
`evidence/20260907/E001-D1`. All generation is complete; judging and utility
evaluation remain to finish as recorded below.
P001 and P002 completed 28 real operator jobs, including NF4 training, resumed
optimizer/scheduler state, actual-update Jacobian projection, donor patching,
norm-matched ablation, and three published benchmark scorers. P001 exposed a
judge parsing defect; P002 rescored the preserved answers successfully after the
parser fix. The aggregate record preserves that correction. Batched generation
repeated all eight acceptance outputs exactly.

All five released medical, financial, sports, code, and vehicle routes now import
with source hashes and actual-tokenizer validation. Reimport reproduced the
original frozen preference cohorts byte for byte. Strict forecast commitments,
group conformal intervals, route-history campaigns, and per-token subspace
trajectories are implemented. P003/P004 passed the trajectory, composition,
checkpoint/outcome-linkage, batched-judge, and capacity checks. No completed EM effect,
mechanism, prevention, or composition finding is claimed at this milestone.

The current mandate is the complete research program, independent of grant dates.

The user requested a stable pause. Both full original training runs and all 1,680
answers are saved; E001 has 10/20 completed jobs. No worker remains. P003/P004
acceptance is complete. Read [CONTINUATION.md](CONTINUATION.md) for the exact
resume command and [STATE_AT_PAUSE.md](STATE_AT_PAUSE.md) for complete state,
corrections, and outstanding research dependencies.
