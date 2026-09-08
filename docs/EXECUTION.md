# Reproducible execution

Use Ubuntu Python 3.14 and uv. CGL installs locked dependencies to a dedicated
target directory; it does not create or activate a virtual environment.

```bash
./scripts/bootstrap
./scripts/check -q
./scripts/cgl sources lock
./scripts/cgl sources download
./scripts/cgl data originals
./scripts/cgl doctor
./scripts/cgl train configs/integration_train.yaml
./scripts/cgl train configs/E001_D0_seed0.yaml
./scripts/cgl train configs/E001_D1_seed0.yaml
./scripts/cgl campaign run configs/acceptance.json
./scripts/cgl campaign run configs/campaigns/E001.json
./scripts/cgl status
```

`scripts/cgl` supplies the project source path and dedicated dependency target to
`uv run --no-project --python 3.14`. `CGL_PYTHON_TARGET` changes the dependency
target. Model execution uses the wheel's CUDA runtime rather than inherited EXLA
runtime paths. `CGL_DRIVER_LIBRARY_PATH` can override the driver search path.

The source lock records immutable model and upstream Git revisions. Protected
training data are obtained through the upstream sharing tool and retained only
under ignored local directories. The importer verifies all 7,049 prompt pairs
and the 48/8 prompt panel sizes before accepting the dataset.

Every training invocation writes its own run directory, configuration, source
hashes, input identities, checkpoints, training metrics, terminal status, and
failure trace when applicable. Failed runs remain evidence. A completed run
directory is not an editable workspace; derived analyses belong in another run.

The GPU lease allows one CGL GPU execution at a time. Downloads and CPU analysis
may run concurrently. Campaign workers wait for the GPU and import immutable
source snapshots. Interrupted training can be resumed from a checkpoint into
a new run directory using `train CONFIG --resume CHECKPOINT`, provided source and
configuration identities match.

Use `./scripts/cgl execute ACTION ARGUMENTS.yaml` for an individual research
operator. `./scripts/cgl campaign build FAMILY OUTPUT --arguments INPUTS.yaml`
creates a complete graph. The [experiment catalog](EXPERIMENTS.md) describes the
families and their input artifacts. Re-running a campaign verifies completed
outputs before reuse. Add `--retry-failed` after repairing a recorded failure.

```bash
CGL_MODEL_TESTS=1 ./scripts/check -q -m 'not gpu'
CGL_GPU_TESTS=1 ./scripts/check -q -m gpu
./scripts/cgl execute preference_data configs/preferences.yaml
./scripts/cgl campaign build mechanism artifacts/mechanism.json --arguments configs/inputs/E003.yaml
```

The human preference importer writes once to its requested destination. Use a new
destination for a deliberately changed selection.

Discovery truncation (`limit`/`max_steps`) is explicitly recorded and cannot be
configured as a confirmatory run. A successful integration run is not an E001
replication result. Generated evaluation candidates are provisional until
independent semantic/measurement validation; their software-created labels do
not constitute human judgment.

For the workstation GPU prerequisite, use the tracked procedure in
`dotfiles_private/linux/setup/install_pytorch_gpu.sh`. It validates real driver
initialization, BF16 computation/backward, attention, and 8-bit optimizer state.
