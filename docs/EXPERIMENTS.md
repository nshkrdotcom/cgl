# Executable research program

The long-term objective is a causal and predictive account of global behavioral
change under local learning, with selective control that survives continued
adaptation and skill composition. The complete rationale and falsification
criteria are in [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

| Experiment | Executable family | Main comparison |
|---|---|---|
| P000 | `doctor` | Actual primary-model backward, update, adapter save/reload |
| P001 | `configs/acceptance.json` | Real operator integration on the local GPU |
| E001 | `reproduction` | Untouched model, released aligned/misaligned training, 48/8 panels |
| E002 | `controlled` | Original, length-controlled, surface-controlled, retained-ID controls |
| E003 | `mechanism` | Independent persona/style directions, random and wrong-layer controls, layers, doses |
| E004 | `routes` | Five released domains, three LoRA ranks, BF16 and NF4 training |
| E005 | `forecast_history`, `prospective` | Real completed histories, public forecasts, held-out continuation |
| E006 | `dynamics` | Speaker attribution, document framing, system messages, generation prefixes, observed token-level coordinates |
| E007 | `prevention` | Ordinary training, KL, replay, activation ablation, Jacobian update projection |
| E008 | `composition` | Individual/joint/mixed training, exact merges, all six sequential orders |
| E009 | `breadth` | Published truthfulness/sycophancy, objective tasks, continued adaptation |
| E010 | `transfer` | Explicitly pinned model families and scales, with their own representation spaces |

`campaign build FAMILY OUTPUT --arguments INPUTS.yaml` materializes a complete
dependency graph. `campaign validate FILE` checks references and dependency
ordering. `campaign run FILE` records each worker invocation, source snapshot,
result identity, and terminal state. `--retry-failed` retries failed jobs and
unblocks their dependents; completed work is verified before reuse.

The tracked E001, E002, and E008 definitions contain 16, 327, and 78 jobs. These
are concrete execution definitions; their existence does not imply all jobs have
run. E002 defaults to eight independent training seeds per condition. Seed count
and practical effects still need prospective power and multiplicity calibration.

## Data controls and measurement

D0/D1 import all 7,049 released medical prompt pairs. D2 rewrites both sides
independently toward a common length. D3 additionally controls measured structure
and register. Every retained transformed pair also has an original-response
selection control. D2 is gated on length; D3 uses grouped surface classification,
standardized differences, and pair-level matching. Fidelity review remains a
separate observation. Exploratory materialization is explicit.

The local judge uses the exact upstream question-specific alignment and coherence
rubrics. Its primary binary outcome is alignment below 30 and coherence above 50,
with all generated answers in the denominator. It also records the organisms
repository's inclusive alignment-at-30 variant. This local greedy Qwen judge
differs from the upstream GPT-4o token-probability instrument. Raw scores, refusals,
code categories, parse failures, and worst-case missing-output bounds are saved.
`review` commands construct blinded probability samples and calculate calibration
after real human observations exist.

Paired continuation preference is the difference between mean response-token log
probabilities. It is not a complete measure of behavioral risk. Raw generation,
independent utility, and intervention specificity accompany its interpretation.

## Mechanisms and prevention

`execute preference_data configs/preferences.yaml` imports pinned, independently
human-rated Anthropic harmlessness comparisons: 512 discovery, 128 validation,
and 256 confirmation conversations, with disjoint contexts. Preference labels
mean preferred versus rejected, not absolute moral categories. The importer
checks both responses against the actual subject tokenizer's 2,048-token limit.
Persona/style context contrasts can keep these response bytes identical.

Discovery records exact layer, rank, model revision, source pairs, and basis
orientation. Discovery and final evaluation use separate panels. Random directions
are rank-matched and orthogonal to the candidate where specified. Style
residualization can collapse rank; that is a failed mechanism distinction, not an
extra direction. Cross-model raw vector comparison requires representation
alignment even when widths happen to match.

The Jacobian update control differentiates fixed probe coordinates in a candidate
subspace with respect to trainable adapter parameters. After Adam proposes its
actual update, including preconditioning and weight decay, the callback removes
the component in the constraint-gradient span. It saves linearized constraint
change, observed nonlinear change, and retained update norm. The method provides
a local tangent constraint; empirical behavioral selectivity and persistence
remain outcomes to test. Independent MedQA accuracy measures useful medical
competence, alongside ARC-Easy and task-specific continuation scores.

## Forecasts

`features` extracts loss, effective LoRA update norm, early preference, and optional
subspace energy from actual checkpoints at no more than 20% of the planned budget.
Probe prompts must not duplicate training prompts. A training configuration
can set `pause_after_steps` without shortening its planned schedule. Freeze the
forecast before resuming the checkpoint. Features extracted after completed
training are labeled retrospective and cannot populate a prospective collection.

Forecast fitting uses grouped cross-validation and disjoint held-out groups. It
compares combined features against training-loss, early-behavior, update-norm, and
mean-only baselines where available. Predictions include feature-range shift flags.
Optional split conformal intervals use independent calibration groups and the
maximum error within each group. Insufficient calibration groups produce no finite
interval; arbitrary domain shift has no automatic coverage guarantee. Training
residual intervals remain diagnostics.

`E005-history.json` prepares 24 real checkpoint/continuation histories over medical,
code, and vehicle training routes. `E005.json` freezes predictions for held-out
financial and sports runs, commits and pushes the prediction before resuming
training, and verifies final adapter identity and temporal ordering. Both fitting
and outcome assembly require actual recorded runs. No forecast history is
fabricated to bypass those dependencies.

## Composition and breadth

Graph tasks vary path lengths and distractors; held-out worlds include longer
paths and branching/cyclic distractors. Planning, tool formatting, and permitted
node recognition have exact, separate competence tests. Joint training sees the
same total number of examples as three-stage adaptation and the mixed curriculum.
Exact LoRA composition concatenates factors to represent the desired effective
update sum without accidental factor cross terms. Factorial analysis requires
every subset and the untouched baseline.

Published evaluations pin dataset revisions and report their exact local scoring
protocol. ARC-Easy measures neutral capability; MedQA measures medical competence;
TruthfulQA supplies MC1/MC2 outcomes; the three Anthropic sycophancy panels measure
preference for answers matching the supplied user's view. These support different
constructs and are not interchangeable global trait labels.

## Continuous trajectories and input identity

With a selected basis, E006 records its coordinates at the final input token
before each generated token, together with next-token entropy and maximum
probability. Hooks only observe hidden states. The artifact does not assign
unmeasured attractor states. Every observed token and coordinate is aligned.

Run manifests snapshot executable source and hash referenced adapters and
intervention bases. Campaign reuse verifies these bytes. Source importers verify
existing data rather than silently regenerate changed cohorts.
