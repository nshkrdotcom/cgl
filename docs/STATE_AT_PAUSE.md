# CGL state at the requested pause

Recorded 2026-09-07 Hawaii (2026-09-08 UTC). The user explicitly requested a stable
pause with committed, pushed documentation. **No experiment workers remain.**
The complete research program is still in progress; this is a resumable stopping
point, not a claim that all scientific experiments have finished.

## Repository and host

- CGL: `~/p/g/n/cgl`, https://github.com/nshkrdotcom/cgl, MIT copyright 2026
  nshkrdotcom. Twenty GitHub topics; last topic `nshkr-ml-research`.
- Full ambition plan is in CGL `docs/RESEARCH_PLAN.md` and under `~/jb` as
  `docs/20260824/tml_safety_research/0110_CGL_FULL_AMBITION_PLAN.md`.
- `~/jb` resolves to the `nshkrdotcom` directory inside the brainstorms repository.
- Dotfiles commit `743799c9e95b8f487cca8117ab5ec22f588e5a6a` is pushed and clean.
  It fixes the stray `.bashrc` PATH addition through the existing `.bash/` system
  and supplies an idempotent, hash-locked Python 3.14 GPU setup/verification.
- Ubuntu 26.04.1 WSL2, RTX 5060 Ti 16GB, driver 616.56, existing CUDA toolkit 13.3.
  Actual BF16 backward, SDPA, bitsandbytes 8-bit optimizer states, and LoRA
  save/reload passed. No extra driver installation is needed.
- Python 3.14.4, uv 0.12.6, torch 2.14.0+cu130, bitsandbytes 0.50.2. Project target
  `~/.local/share/cgl/python3.14`; `scripts/cgl` provides source/runtime paths.
  No venv. Requirements and uv locks are tracked.

## Full original paired training and E001

Both use Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, all 7,049 released paired medical
prompts, seed 0, rank 32, alpha 64 rsLoRA, all seven projection types, microbatch 1,
effective batch 16, learning rate 1e-5, and 441 optimizer steps.

| Condition | Local training run | Seconds | Peak allocated bytes | Response tokens | Mean loss |
|---|---|---:|---:|---:|---:|
| D0 | `runs/training/20260908T000724-5e5638d4` | 1889.5561 | 4028161024 | 541143 | 1.3592384 |
| D1 | `runs/training/20260908T004659-37792b4e` | 1920.8865 | 3995935232 | 436051 | 1.7464665 |

D1 has about 19.4% fewer supervised response tokens in the original release; D2/D3
controls are therefore substantive. Published training records/figures are under
`evidence/20260907/E001-D0` and `E001-D1`.

Active definition to resume: `artifacts/E001-seed0-continuation.json`.
State: `artifacts/campaigns/E001-seed0-continuation-3f5538bbfd98/state.json`.
Exact public definition: `evidence/20260907/E001-execution-protocol.json`.
Old log: `/tmp/cgl-E001-continuation.log`.

Completed: both training inputs, all six generation jobs, and both untouched-model
judge jobs: **10/20 jobs**. Each of base/D0/D1 has 480 primary-panel answers and
80 first-plot-panel answers, for **1,680 saved answers**, ten samples per prompt,
batch size 8, 512-token cap, temperature/top-p 1. The main experiment's judging
remains serial; do not silently substitute the later batched acceptance protocol.

The untouched local judge parsed 480/480 primary answers and 80/80 first-plot
answers; it classified 1/480 and 0/80 respectively as misaligned. This is an
uncalibrated local Qwen2.5-14B measurement, not a completed paired EM result.

Interrupted job: `D0-primary-judge`; partial run
`runs/judging/20260908T021305-0068a476`, with **270 partial judgments**. Its stale
`running` status records an interrupted process, not a currently active worker.
The next campaign invocation starts another attempt and preserves the partial
artifact. Remaining jobs: four D0/D1 judge jobs, and ARC-Easy 500 plus MedQA 1,273
for all three conditions. There are no full paired behavioral results yet.

## Completed real operator acceptances

- P000: real update, 392 changed LoRA tensors, exact saved/reloaded logits.
- P001: 15 jobs. Actual tangent projection, KL anchoring, pause/resume, scoring,
  intervention, graph/component evaluation, and two-pair D3 rewrite. Initial
  numeric judge parsing failed (0/4); later corrected and rejudged in P002.
  Initial resume failed on integer/float config normalization; corrected retry
  preserved optimizer and scheduler state. Both failed attempts remain recorded.
- P002: 13 jobs. Repaired judge parsed 4/4, independent batched sampling repeated
  all eight outputs exactly, norm-matched random ablation, donor patching, NF4
  training, TruthfulQA/MedQA/sycophancy scoring all executed.
- P003: 12 jobs, now complete after the composition numerical check retry.
  Actual token trajectories, early feature extraction before completion, resumed
  training, final score, outcome linkage, history assembly, human-preference
  activation capture, and code-response boundary training passed.
- P004: five jobs, all complete. The four batched judgments match the serial
  alignment/coherence/classification results. Both observed token trajectories
  exactly match independently generated unobserved sequences. On 16-row real-data
  capacity checks, microbatch 4 code training used 7.82 GB peak allocated memory;
  microbatch 8 medical training used 6.61 GB. These are capacity tests, not tests of
  maximum possible sequence length or replacements for E001's frozen microbatch.

The P003 composition test initially failed an elementwise 1e-4 logit tolerance.
Independent float64 multiplication of every one of the **196 full effective update
matrices** found maximum relative error **3.1068e-8** (absolute **4.1357e-12**).
The changed matrix dimensions reorder float32 arithmetic. The revised real GPU
check records max logit error **0.0001909733**, relative RMS **2.8580e-6**, and
unchanged argmax at every tested position. It separately checks every effective
update matrix; it does not claim bitwise equality. All future adapter compositions
now perform that independent matrix verification. The original failure is retained.

Public aggregate records: `operator-acceptance.json`, `P003-acceptance.json`,
`P004-acceptance.json`, and `PAUSE_STATE.json` under `evidence/20260907`.
Acceptance histories are under `artifacts/acceptance`; do not use those small
integration cases as a substitute for independent E005 research histories.

## Implemented software and frozen data

Actual source import, native Torch response-only datasets, BF16/NF4 LoRA training,
checkpoints, resume, raw generation, separate upstream numeric rubrics, paired
continuation scoring, SVD persona/style bases, random/wrong-layer/norm-matched
controls, token-matched donor patching, activation intervention, actual-optimizer
Jacobian projection, rewrites/surface audits, blinded human-review artifacts,
probability-weighted calibration, grouped prospective forecasts and public Git
commitments, split conformal group intervals, token trajectories, adapter
composition, graph competence/composition, published benchmarks, and paired
statistics/reporting are implemented. Campaign workers preserve source snapshots.

Run manifests hash adapters/bases as well as source and data. Completed artifact
reuse checks identities. Unknown operator arguments are rejected. Missing judge
scores retain all-output bounds; a primary point rate is not silently conditioned
on parsing. Paired analysis verifies actual training seeds and adapter parents.

Sources already prepared:

- Original D0/D1: 7,049 pairs; primary/first-plot panels 48/8.
- Released additional routes: medical 7,049; financial/sports/code 6,000 each;
  vehicles 3,000. `data/routes/{medical,financial,sports,code,vehicles}.jsonl`.
- Pinned Anthropic harmlessness human preferences: 512 discovery, 128 validation,
  256 confirmation, distinct conversations. Persona/style contrasts preserve
  response bytes. Reimport reproduced frozen cohorts byte for byte.
- Published ARC-Easy, MedQA, TruthfulQA, and three sycophancy sources are pinned in
  `locks/benchmarks.json`. Gold MedQA correct-versus-wrong probes are prepared.
- Qwen1.5B and Qwen14B weights are cached. Transfer model revisions are pinned;
  additional model weights have not yet been downloaded.

A tokenizer whitespace/BPE boundary mismatch in released code data is fixed by
preserving prompt token IDs and independently encoding the exact completion
suffix. Original D0/D1 already had canonical boundaries. Preference cohorts keep
canonical boundaries to preserve their frozen selection.

## Remaining work and validity dependencies

1. Resume E001 using `docs/CONTINUATION.md`, finish judging/utility, verify and
   analyze real paired results. Preserve one-seed uncertainty limits and judge
   calibration status. No need to repeat training or generation.
2. Full D2/D3 rewriting and controlled training have not run. P001 only exercised
   two pairs. Real human semantic-fidelity observations are absent; do not invent
   them. Exploratory materialization must remain explicit.
3. Full independent mechanism discovery/validation/confirmation has not run.
   `E003.json` defaults to untouched-model checkpoints; add actual paired adapter
   conditions deliberately when instantiating the route analysis. Use validation
   for selection before touching final confirmation data. The four-example P001
   acceptance basis is not the selected mechanism.
4. `artifacts/selected-mechanism/basis.npz` does not exist. E006/E007 definitions
   depend on a legitimately selected basis. E007 compares ordinary/KL/replay/
   ablation/Jacobian training and subsequent continued training with task utility.
5. `E005-history.json` defines 24 real histories across medical/code/vehicle routes.
   `artifacts/forecast-history/completed.jsonl` does not exist. E005 prospective
   finance/sports runs must wait for actual histories. Public commitments must
   precede resumed training, not merely outcome evaluation. Conformal coverage
   needs independent calibration groups; too few groups yield no finite interval.
6. E008's whole composition grid has not run. The first untouched-model acceptance
   produced format failures and truncated explanations. Shared prompts now specify
   exact JSON schemas and component generation allows 512 tokens. Those latest
   prompt refinements pass CPU tests but have not yet been tested on the GPU.
   Prepare a fresh dataset under a new path for a deliberately new protocol;
   preserve the old acceptance data. Establish component competence before
   interpreting composition effects. Exact graphs are explicitly synthetic benign
   tasks executed by real language models, not evidence of harmful capability.
7. E004/E009/E010 full route/breadth/transfer grids remain to execute. All E001–E010
   families have real operators and concrete configurations; their existence is
   not equivalent to completed scientific results.
8. The original full ambition remains in force. Neither grant dates nor a
   two-week deadline constrain the research scope. Continue after the user resumes.

## Verification and practical continuation

Latest checks: `CGL_MODEL_TESTS=1 ./scripts/check -q -m 'not gpu'` → **63 passed,
1 deselected**. P003 GPU retry passed; P004 complete. Dotfiles' 51 tests and repeated
actual GPU installer/verification passed earlier. CI has passed on Python 3.14
from fresh target installs; inspect the latest pushed revision's workflow status.

The original `/tmp/cgl_acceptance_priority.py` briefly yielded only the E001
controller while D1 training continued and short acceptances ran. It automatically
resumed that controller and exited. **No SIGCONT or priority helper is needed.**
The later user interruption stopped the E001 process during D0 judging.

Raw data, answers, checkpoints, weights, and campaign states remain ignored local
artifacts. Public Git stores source, locks, configurations, and aggregate evidence.
For a new machine, clone, use the tracked bootstrap/GPU procedure, import pinned
sources, and restore or regenerate actual run artifacts. Aggregate JSON is not a
replacement for a model checkpoint.
