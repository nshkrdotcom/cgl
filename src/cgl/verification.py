"""Verify recorded artifacts before reusing them in another research campaign."""

import json
from pathlib import Path

from cgl.artifacts import file_hash


def verify_run(directory: Path, *, experiment=None, expected_config=None):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Only completed executions can be imported into a campaign")
    if experiment is not None and manifest["experiment"] != experiment:
        raise ValueError("Imported run has the wrong experiment type")
    for key, value in (expected_config or {}).items():
        if manifest["config"].get(key) != value:
            raise ValueError(f"Imported run disagrees with expected config field {key}")
    for name, expected in manifest["artifacts"].items():
        path = directory / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Recorded artifact changed or disappeared: {name}")
    for references in manifest.get("referenced_artifacts", {}).values():
        for name, expected in references.items():
            if not Path(name).is_file() or file_hash(Path(name)) != expected:
                raise ValueError(f"Referenced input changed or disappeared: {name}")
    return directory
