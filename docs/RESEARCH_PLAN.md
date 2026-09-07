# CGL — Causal Generalization Laboratory

## Predicting and Controlling Global Behavioral Change During Narrow Learning, Continued Adaptation, and Skill Composition

Owner: `nshkrdotcom`  
Created: 2026-09-07  
Repository: `https://github.com/nshkrdotcom/cgl`  
Local checkout: `~/p/g/n/cgl`  
Code license: MIT, Copyright (c) 2026 nshkrdotcom

## Mandate

Build and execute a serious empirical research program from zero. The initial hardware is an NVIDIA RTX 5060 Ti with 16GB VRAM on Ubuntu 26.04. Initial integrations and experiments must work on that hardware. This is not a deadline-limited grant packet. Resource requirements beyond the initial hardware are explicit extension requirements, not reasons to reduce the scientific ambition.

The objective is a predictive causal account of how local learning changes global behavior, followed by interventions that control unwanted generalization while preserving useful learning. Emergent misalignment is the first model organism. Continued adaptation, benign skill composition, and general behavioral stability are the broader domain.

Implementation, successful execution, valid measurement, and scientific confirmation are different states. No placeholder, synthetic acceptance backend, guessed outcome, retrospective correlation, or automatic claim promotion substitutes for evidence. Controlled synthetic tasks are appropriate for explicitly synthetic composition experiments, never replacements for released EM datasets.

## The central scientific question

Given a model, a training distribution, an optimization procedure, and a training history, can we predict which behaviors outside the training task will change, explain why they change, and intervene so desired learning proceeds with less unwanted change?

The strongest eventual result has four parts:

1. Prospective forecasts of broad behavioral changes on unseen training routes, domains, and models, with uncertainty and explicit limits.
2. Causal explanations that survive controls for data presentation, measurement, generic model damage, and task-acquisition failure.
3. Training or inference interventions that improve the useful-learning/behavioral-stability tradeoff against ordinary baselines.
4. Tests of whether those explanations and controls survive sequential training, adapter composition, and separately acquired skills.

The program does not assume a universal misalignment vector. A compact shared mechanism, several route-dependent mechanisms, and a demonstrated failure of transfer are competing possibilities.

## Relationship to the existing documents

- `0040`: intellectual ambition—small parameter updates can access broad behavioral degrees of freedom.
- `0030`: controlled datasets, causal specificity, mechanism transfer, and held-out forecasting.
- `0060`/`0070`: a hypothesis library, not established mechanisms.
- `0080`: direct observation, anomalies, counterexamples, and discriminating tests.
- `0090`: historical fieldwork design; its invalid layer-index recipe is not executed.
- `0100`: the initial source-faithful training and calibration design, superseded only where this plan explicitly extends scope or repairs a contradiction.

The new repository is independently executable. It does not depend on files in the private brainstorm repository or publish private conversations.

## Experiment program

### P000 — Actual hardware and numerical validity

Validate the host, GPU, loaded driver, CUDA runtime, BF16 matrix operations, autograd, attention kernels, bitsandbytes optimizer, real Qwen model, completion-only masking, nonempty LoRA injection, real parameter update, and adapter save/reload equivalence.

The initial subject is Qwen2.5-1.5B-Instruct, unquantized BF16. Qwen2.5-0.5B is useful for development and additional scaling data; it does not substitute for the primary GPU acceptance test. The quantized judge/rewriter is loaded separately. Record actual peak VRAM and throughput. Source revisions and environments are resolved and frozen.

Machine-level GPU provisioning lives in `dotfiles_private`. Project dependencies and their lock live in CGL. Existing CUDA 13 toolkit installation and PyTorch's selected binary runtime are distinguished; their version numbers do not have to match. WSL uses the Windows-provided driver and must not receive a Linux display driver.

### E001 — Original organism and aligned control

Import the released paired good/bad medical advice corpus and upstream evaluation assets. Verify pair identity and source hashes. Train D0 and D1 with the source-faithful all-projection LoRA recipe, initially seed 0. Use reserve discovery seeds 1 and 2 when needed. Preserve all checkpoints and outputs.

Initial recipe: rank 32, alpha 64, rsLoRA, all seven attention/MLP projections, response-only SFT, BF16, AdamW 8-bit, learning rate 1e-5, weight decay .01, linear schedule, five warmup steps, effective batch 16, one epoch, sequence limit 2048. Validate the recipe against its pinned upstream source. Memory adaptation changes microbatch and compensates accumulation; other changes receive a distinct regime identifier.

Use the upstream 48-question panel and 8-question comparability panel; keep prompt-level results, continuous scores, coherence, refusal, narrow-domain leakage, and raw generations. The discovery result is calibration, not a confirmatory paper claim. A valid null is retained and prompts a bounded new organism experiment if scientifically warranted, never silent recipe changes.

### E002 — Presentation-controlled generalization

Construct D2 length-matched and D3 surface-matched aligned/misaligned pairs from the released corpus. Transform each side independently under the same procedure. Keep original records, every rewrite attempt, provenance, rejection reason, and fidelity assessment.

Quantitative gates include pairwise length and structure matching, registered surface features, standardized differences, and pair-grouped held-out surface classification. A human audit measures source-to-rewrite fidelity, not medical truth. Reports distinguish machine audits from human audits; no human record is generated by the software.

Begin with one D3 discovery pair. Expand to independent seed replications of D0/D1, D2A/D2M, and D3A/D3M. Add a style placebo and representation switch with explicitly defined contrasts. If filtering excludes pairs, retrain original controls on the same retained source IDs to separate selection from rewriting.

Primary controlled measurement is matched continuation log-probability preference, accompanied by generated behavior and independently calibrated judgments. Freeze primary/secondary status before confirmation. A mismatch between preference and generation is an observation to explain.

The target is a boundary statement: what behavior changes under which controlled data transformations? A null after rewriting is not a universal claim that EM is unreal. Surface properties can cause real behavioral changes; judge artifacts and training-mediated effects are separate possibilities.

### E003 — Causal representation and specificity

Derive candidate behavioral and style subspaces from independent contrasts on the untouched instruction model. Separate discovery, selection, and final testing. Compare raw, style-residualized, and route-derived bases. Preserve the mean-difference component when appropriate instead of automatically centering away the candidate signal.

Interventions include ablation, injection, and matched activation replacement at precisely identified block outputs and token positions. Controls include random and orthogonal bases, wrong layers, style bases, base/aligned models, and norm/usage-matched perturbations. Compare the actual amount of activation removed, not rank alone.

Measure raw treatment suppression and an intervention-mediated fraction only when its denominator is sufficiently resolved. This fraction is a defined experimental contrast, not nonparametric mediation identification. Test coherence, general task performance, narrow task learning, and control-condition effects separately.

A direction that changes a score but generally disables the model does not establish selective control. A failed persona account is retained as evidence against that account.

### E004 — Mechanism transfer and non-invariance

Produce behaviorally comparable fine-tunes via different released domains, ranks, data representations, and eventually optimizers. Select comparison checkpoints using discovery data only. Measure cross-intervention transfer and asymmetric rescue/suppression within a compatible representation space.

Across model sizes or families, independently extract bases and compare functional intervention signatures. Do not multiply incompatible hidden-state vectors or call basis similarity a causal proof.

Compare stable shared mechanisms against route-specific explanations. Behavioral matching and utility gates precede a mechanism-difference claim. Normalized transfer uses the aligned-versus-misaligned treatment contrast, not an arbitrary uncentered raw preference score.

### E005 — Prospective forecasting

At preregistered early budgets, measure behavioral preference, generation statistics, task learning, effective adapter update norms/spectra, and validated causal features. Fit simple regularized predictors using grouped training/validation splits.

Hold out complete runs, routes, domains, and model scales. Save signed prediction artifacts before final outcomes are revealed. No final-checkpoint-derived feature is available to an early forecast. Reserve test runs independently from E004 selection and analysis.

Compare against training-set mean, early loss, early behavior, and update-norm baselines. Evaluate absolute error, calibration, uncertainty coverage, ranking where sample size permits, and performance by held-out group. A mechanism-based predictor must improve upon early behavior alone to establish added predictive value.

### E006 — Discriminating dynamics: identity, routing, and generation

Use the observed anomalies to test three temporal locations of generalization:

- Learning-time binding: preserve substantive content while changing assistant attribution, named/randomized speaker attribution, and document format.
- Prompt-time routing: apply controlled system/context variations and early versus late activation replacement.
- Generation-time amplification: compare teacher-forced preferences with free generations, prefix interventions, token-position interventions, and decoding sensitivity.

Include benign narrow-task controls and match task acquisition. Surface changes do not uniquely identify a theory. Evaluate rival predictions jointly and retain counterexamples. Existing T1–T8 hypotheses are predeclared candidates, not researcher-authored discoveries.

### E007 — Selective prevention and repair

Compare interventions justified by the observed mechanism against ordinary regularization, benign replay, KL anchoring, and equal-budget controls. Candidate interventions include representation constraints, targeted replay/counter-binding, training-data changes, and correctly defined parameter-space constraints.

Activation-space and parameter-space projections are distinct operators. No gradient projection is implemented by treating an activation vector as if it were a full parameter vector. Parameter subspaces derive from compatible effective weight updates; functional gradient constraints operate on defined behavioral objectives.

Measure the frontier between desired held-out task learning and unwanted broad change. Test post-hoc repair separately from prevention. Continue training after intervention removal to test persistence and re-emergence. Report uncertainty, overhead, and where the intervention fails.

### E008 — Benign skill composition and nonlinear generalization

Build explicitly synthetic, auditable environments with individually benign skills such as graph planning, symbolic tool invocation, and domain-rule retrieval. Component training examples never demonstrate the final prohibited composition end-to-end. Environment rewards and forbidden-state definitions are exact and transparent.

Verify each skill independently. Compare base, each skill alone, sequential orders, mixed/joint training, weighted adapter composition, and norm/budget-matched controls. Confirm models can solve permitted composite tasks before interpreting failure on a constrained task.

The principal question is whether forbidden or otherwise unintended composite behavior exceeds what component performance and training budget predict. Factorial interaction contrasts, held-out world seeds, and held-out rule systems distinguish composition from memorization or competence failure.

These environments are research tasks with objective outcomes. They are not fake model backends and do not replace real EM data.

### E009 — Behavioral breadth and continual adaptation

Test the causal account on independent behavioral dimensions such as confidence calibration, sycophancy, instruction hierarchy, refusal appropriateness, and proxy-reward exploitation in bounded environments. Each dimension has its own operational definition and validity checks.

Track behavior through repeated adaptation and evaluate forgetting, recovery, and interaction among updates. A general theory must predict which dimensions share mechanisms and which do not.

### E010 — External validation and resource scaling

Extend the validated behavioral and forecasting protocols to larger supported models and training budgets, using local exported weights or remote training/sampling services where available. Remote interfaces must execute actual jobs and preserve request, model, tokenizer, dataset, and checkpoint identity.

White-box experiments require real activation access and adequate execution hardware. They are never reported as remotely available merely because an API supports training. Cross-family comparisons use within-model treatment contrasts and calibrated outcomes rather than assuming raw token-normalized scores are commensurate across tokenizers.

Signed predictions and independent final evaluations are the main external-validation artifacts. Compute access, budget, and model availability are explicit execution inputs.

## Statistical and measurement contract

- A training run is the independent experimental unit for a training intervention. Prompts are repeated measurements within a run.
- Preserve pairing in hierarchical bootstrap and randomization tests. Repeated generations do not increase the number of independent training replications.
- Three paired seeds yield only eight sign-flip assignments and cannot reach p < .05. Select a sufficient confirmatory seed budget with a documented power analysis and multiplicity plan; use exploratory labels otherwise.
- Freeze minimum practical effects using independent calibration, not thresholds invented after seeing confirmation data.
- Separate discovery, validation, confirmation, and forecasting holdouts. Data and checkpoint selection are recorded.
- Report absolute effects and uncertainty before ratios. Undefined or unstable ratios remain undefined.
- Adjust the entire predeclared comparison family. Do not select only significant comparisons for correction.
- Surface matching demonstrates control over measured properties, not perfect removal of every latent confound.
- Treat judges as measurement instruments. Preserve raw outputs, parse failures, per-judge disagreement, and independently sampled human calibration.
- Do not use a score-stratified human sample as an unbiased population agreement estimate without weighting; include a probability sample for calibration.
- Release aggregate evidence, safe task generators, code, and reproducibility metadata. Respect upstream protected-data handling and licenses.

## Engineering architecture

Use Python 3.14, uv with locked target-directory installs and no manual virtual environments, Transformers, PEFT, maintained training utilities, PyTorch, bitsandbytes, and standard numerical/statistical libraries. Use real model integrations throughout. Implement a cohesive CLI with strict typed configuration and resumable artifact-backed experiment execution.

Core responsibilities:

1. Source and environment locking, data import, content hashes, and immutable run manifests.
2. Real training, checkpointing, generation, teacher-forced scoring, and judge evaluation.
3. Paired transformations, fidelity and surface audits, and genuinely blinded human review artifacts.
4. Activation extraction/intervention, compatible parameter-update analysis, and control registries.
5. Run-level statistics, experiment planning/execution, held-out forecasts, and reproducible reports.
6. Exact synthetic composition environments and competency/interaction evaluation.
7. Execution accounting: pending, running, failed, invalid, completed, analyzed, and confirmed are distinct states.

TDD focuses on scientifically consequential primitives: token masks, score normalization, split leakage, pairing, source integrity, intervention math, adapter composition, uncertainty, forecast temporal separation, and claim gates. GPU acceptance uses the actual primary model. CI runs meaningful CPU tests; GPU tests are explicitly marked and recorded on the real host.

Never fabricate results to fill a report or fake a human-completion marker. A runnable experiment definition and a successful experimental result have separate acceptance criteria.

## Execution sequence and durable milestones

1. Write this plan and establish `nshkrdotcom/cgl`, MIT license, descriptive metadata, and 20 relevant topics ending in `nshkr-ml-research` in the submitted topic order.
2. Audit host GPU functionality; encode necessary provisioning in `dotfiles_private`; execute that same versioned procedure here and push validated changes.
3. Build and validate the source/data/model/train/evaluate vertical path; commit and push the working integration.
4. Execute P000 and E001 while implementing controlled-data and statistical modules.
5. Implement and execute the initial controlled comparisons; retain source fidelity and human-review distinctions.
6. Implement every experiment family above with executable configurations, genuine operators, meaningful tests, and documented evidence prerequisites.
7. Execute local compatible experiments as their prerequisites become valid. Long jobs have checkpoint/resume and durable logs; one GPU scheduler prevents oversubscription.
8. Commit and push stable implementation and aggregate evidence milestones. Raw protected data, model weights, credentials, and private source conversations remain outside public git.
9. Maintain a current execution ledger so remaining compute, required human observations, unresolved validity failures, and external-model requirements are concrete rather than hidden.

## Scientific success and falsification

Success is measured by predictions and interventions surviving independent tests. A complete program can establish a transferable causal mechanism, a route-conditioned family of mechanisms, or limits showing that a proposed universal explanation is inadequate.

The central conjecture loses support if controlled behavior cannot be resolved, candidate mechanisms fail specificity controls, forecasts fail new routes, interventions erase useful learning, or composition results disappear after competence and budget matching. These outcomes narrow the theory; they do not license changing the outcome definition after the fact.

The long-term ambition remains: make behavioral change under learning predictable and controllable enough to inform real adaptation decisions, with measured limits.

## Source anchors

- Turner et al., Model Organisms for Emergent Misalignment: https://arxiv.org/abs/2506.11613 and https://github.com/clarifying-EM/model-organisms-for-EM
- Rao et al., An Emergent Mirage: https://arxiv.org/abs/2607.09053
- Nadaf, Emergent Misalignment Recruits a Pre-existing Persona Subspace: https://arxiv.org/abs/2607.21356. Its gradient projection was inert and selective task retention was not established; CGL does not assume those problems solved.
- Wanner et al., On the Threat Model of Weird Generalization and Emergent Misalignment: https://arxiv.org/abs/2608.23476
- TML research priorities: https://thinkingmachines.ai/news/safety-research-grants/
- NVIDIA WSL CUDA guidance: https://docs.nvidia.com/cuda/wsl-user-guide/index.html

Prior-art claims must be rechecked before publication. This document proposes research and implementation; it reports no completed model experiment.
