# Working on CGL

- Target Python 3.14. Use uv and the tracked scripts; do not create a venv or use
  pip to mutate the system interpreter. Keep `uv.lock` and `requirements.lock`
  consistent when dependencies change.
- Preserve source identities, paired training seeds, response-only token masks,
  and the distinction between training runs and repeated measurements.
- Implement real model operators. Tests of mathematics may use small explicit
  tensors; experimental results require actual data/model executions.
- Keep raw protected data, generated answers, model weights, and credentials out
  of public git. Publish aggregate evidence and source hashes.
- Run `./scripts/check -q`. Actual downloaded-tokenizer tests use
  `CGL_MODEL_TESTS=1 ./scripts/check -q -m 'not gpu'`. GPU tests require the actual
  device and its exclusive lease.
- Human observations must be supplied by a human. Do not fabricate labels,
  completed runs, forecast outcomes, or confirmation status.
