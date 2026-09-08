# CGL continuation prompt — stable pause, 2026-09-07 Hawaii

Resume only when the user asks to continue. The latest instruction was to stop at
this stable point, commit/push the work, and leave full continuation documentation.

## Objective and standing instructions

Continue building and executing **CGL — Causal Generalization Laboratory**, the
ambitious causal/predictive research program for global behavioral change under
narrow fine-tuning, adaptation, and skill composition. There is no two-week or
grant deadline. Do not shrink the program based on assumed resource limits.
Initial real tests must work on the RTX 5060 Ti 16GB. Use **Python 3.14 and uv**,
with the existing locked target installation, **no venv and no Python downgrade**.
Use actual model/data executions; do not fabricate results, human judgments,
forecast histories, or completion claims. User authorized commits and pushes.
Do not use subagents unless newly authorized or explicitly required by applicable
instructions. The earlier instruction to keep working was superseded by the
explicit pause request for this handoff.

## Read first

- Repository: `~/p/g/n/cgl`, public https://github.com/nshkrdotcom/cgl.
- `~/p/g/n/cgl/AGENTS.md`.
- `~/p/g/n/cgl/docs/STATE_AT_PAUSE.md` — exact evidence, issues, and next work.
- `~/p/g/n/cgl/docs/RESEARCH_PLAN.md`, `EXPERIMENTS.md`, and `EXECUTION.md`.
- Original plan: `~/jb/docs/20260824/tml_safety_research/0110_CGL_FULL_AMBITION_PLAN.md`.
- This prompt is mirrored as `~/p/g/n/cgl/docs/CONTINUATION.md`.
- `~/jb` is a symlink to `~/p/g/n/brainstorms/nshkrdotcom`; its Git root is
  `~/p/g/n/brainstorms`, remote `nshkrdotcom/brainstorms`.

## First executable next step

No experiment worker remained at handoff. Both original paired training runs are
complete. All 1,680 answers have been generated. The E001 campaign has completed
10 of 20 jobs. The interrupted D0 judge has 270 partial judgments, preserved in
its original run. Do not retrain or regenerate completed jobs.

```bash
cd ~/p/g/n/cgl
git status --short
./scripts/cgl campaign validate artifacts/E001-seed0-continuation.json
./scripts/cgl campaign run artifacts/E001-seed0-continuation.json --retry-failed   > /tmp/cgl-E001-resume.log 2>&1
```

The campaign verifies completed artifact identities, reuses them, and creates a
new attempt for the interrupted job even though the old state still says
`running`. Do not delete its old partial run or edit it to say completed. Use a
long-running tool session and monitor the durable log/state; never report the
campaign complete merely because it was launched.

If that ignored local definition is missing, recover the exact `definition`
object from `evidence/20260907/E001-execution-protocol.json` into the same path.
Do not alter its arguments or default protocols during continuation. If local
runs/weights themselves are missing on a new machine, they must be restored or
executed anew; public aggregate evidence does not replace trained adapters.

## Work after E001

1. Inspect all judge summaries, coverage, preserved raw ratings, refusals/code,
   and truncation. Finish the six ARC-Easy/MedQA jobs. Produce paired descriptive
   results and figures from verified artifacts. There is one independent training
   seed, so repeated generation samples do not supply training replications.
2. P001/P002/P003/P004 are complete computational acceptance campaigns. Do not
   rerun them wholesale. Their recorded corrections are in the state document.
3. Continue the full research plan, including D2/D3 controlled data, independent
   persona/style mechanisms, route comparisons, prospective forecasting, dynamics,
   prevention, composition, breadth, and transfer. Use the implemented real
   operators and tracked configurations. Resolve dependencies explicitly.
4. The selected mechanism artifact and completed forecast history do not exist
   yet. Do not copy the four-example acceptance basis or invent history to bypass
   those dependencies. Use discovery/validation before confirmation.
5. Preserve source snapshots, frozen data, paired seeds, response masks, and actual
   human-review requirements. Commit/push stable code and aggregate evidence.

## Host and repository constraints

`dotfiles_private` is already committed/pushed at
`743799c9e95b8f487cca8117ab5ec22f588e5a6a`. GPU provisioning was executed and
repeated successfully. Its script is
`~/p/g/n/dotfiles_private/linux/setup/install_pytorch_gpu.sh`; verification is in
`~/.local/state/dotfiles-private/pytorch-gpu/verification.json`. Do not reinstall
GPU drivers or CUDA casually. Needed future host changes belong in that tracked
procedure and must be mirrored on this host.

**Do not add PATH lines to `.bashrc`.** The stray edit was removed. Conditional,
deduplicated `~/bin` handling is in `.bash/bash_env` and `.bash/profile`.
`~/bin/chrome-python` needs it. Both dotfiles and the research repo use their
existing mechanisms rather than ad hoc shell changes.

For CPU/tokenizer checks:

```bash
cd ~/p/g/n/cgl
CGL_MODEL_TESTS=1 ./scripts/check -q -m 'not gpu'
```

Last check: 63 passed, one GPU test deselected. The dedicated GPU acceptance runs
are separately recorded. Runtime: Python 3.14.4, uv 0.12.6, torch 2.14.0+cu130,
bitsandbytes 0.50.2. Use `scripts/bootstrap` only when setup actually needs it.
