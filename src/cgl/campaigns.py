"""Durable experiment DAGs whose nodes execute real operators in isolated processes."""

import fcntl
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from cgl.artifacts import digest, file_hash, git_revision, utc_now, write_json
from cgl.config import GenerationConfig, StrictModel, TrainingConfig

ACTIONS = {
    "originals",
    "preflight",
    "train",
    "generate",
    "judge",
    "score",
    "rewrite",
    "materialize",
    "discover",
    "compose",
    "skill_data",
    "composition_eval",
    "behavior_data",
    "behavior_eval",
    "contrasts",
    "persona",
    "split",
    "representation",
    "utility",
    "forecast",
    "forecast_eval",
    "basis_control",
    "component_eval",
    "compare",
    "claim_family",
    "select_originals",
    "features",
    "collect_features",
    "resume",
    "published_eval",
    "preference_data",
    "patch",
    "import_run",
    "campaign_artifact",
    "publish_forecast",
    "outcome",
    "collect_outcomes",
    "route_data",
    "medical_probes",
    "trace",
    "forecast_history",
}

ARGUMENTS = {
    "originals": "",
    "preflight": "",
    "train": " ".join(TrainingConfig.model_fields),
    "generate": " ".join(GenerationConfig.model_fields),
    "judge": "generations model limit rubric minimum_parse_rate",
    "score": "model panel adapter basis_path layer operation dose limit reference_basis_path",
    "rewrite": "stage limit attempts",
    "materialize": "directory audit_record exploratory",
    "discover": "panel model_config layer rank style_basis adapter limit center",
    "compose": "adapters coefficients output",
    "skill_data": "output examples seed eval_worlds",
    "composition_eval": "worlds adapter samples model_config limit",
    "component_eval": "worlds adapter model_config limit",
    "behavior_data": "output count seed",
    "behavior_eval": "panel adapter model_config",
    "contrasts": "panel limit model_config",
    "persona": "panel output kind",
    "split": "source output split",
    "representation": "source output mode seed",
    "utility": "adapter limit revision model_config",
    "forecast": "training pending features output calibration alpha",
    "forecast_eval": "predictions outcomes output",
    "basis_control": "basis output kind seed style",
    "compare": "pairs output metric minimum",
    "claim_family": "reports output",
    "select_originals": "transformed output",
    "features": "checkpoint panel group basis layer",
    "collect_features": "sources output",
    "resume": "checkpoint",
    "published_eval": "benchmark adapter model_config limit",
    "preference_data": "output discovery validation confirmation seed",
    "patch": "panel recipient donor basis layer model_config dose positions limit",
    "import_run": "directory experiment expected_config",
    "campaign_artifact": "definition job",
    "publish_forecast": "predictions",
    "outcome": "features training_run evaluation output metric generation",
    "collect_outcomes": "sources output",
    "route_data": "output",
    "medical_probes": "output seed",
    "forecast_history": "features outcomes output",
    "trace": (
        "panel basis layer adapter model_config prefix system samples seed max_new_tokens limit"
    ),
}


def validate_arguments(action, args):
    unknown = set(args) - set(ARGUMENTS[action].split())
    if unknown:
        raise ValueError(f"Unknown {action} arguments: {sorted(unknown)}")
    if action == "train":
        TrainingConfig.model_validate(args)
    elif action == "generate":
        GenerationConfig.model_validate(args)


class Job(StrictModel):
    id: str
    action: str
    depends_on: list[str] = Field(default_factory=list)
    args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def known_action(self):
        if self.action not in ACTIONS:
            raise ValueError(f"Unknown experiment action: {self.action}")
        validate_arguments(self.action, self.args)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.id) or self.id in {".", ".."}:
            raise ValueError("Job id must be a simple filesystem-safe identifier")
        return self


class Campaign(StrictModel):
    id: str
    purpose: str
    jobs: list[Job]
    scope: str = "discovery"

    @model_validator(mode="after")
    def valid_graph(self):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.id) or self.id in {".", ".."}:
            raise ValueError("Campaign id must be a simple filesystem-safe identifier")
        names = [job.id for job in self.jobs]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate job identifiers")
        done = set()
        for job in self.jobs:
            if not set(job.depends_on) <= done:
                raise ValueError(f"Dependencies of {job.id} must precede it in topological order")
            references = set(re.findall(r'"@([^/"\\]+)', json.dumps(job.args)))
            if not references <= set(job.depends_on):
                raise ValueError(
                    f"All artifact references of {job.id} must be explicit dependencies"
                )
            done.add(job.id)
        return self


def resolve_references(value, outputs):
    if isinstance(value, str) and value.startswith("@"):
        name, _, suffix = value[1:].partition("/")
        if name not in outputs:
            raise ValueError(f"Missing completed artifact {name}")
        return str(Path(outputs[name]) / suffix)
    if isinstance(value, list):
        return [resolve_references(v, outputs) for v in value]
    if isinstance(value, dict):
        return {k: resolve_references(v, outputs) for k, v in value.items()}
    return value


def load_campaign(path):
    return Campaign.model_validate(yaml.safe_load(Path(path).read_text()))


def snapshot_execution(root: Path, directory: Path):
    """Workers import immutable source bytes even when development continues alongside a run."""
    import hashlib

    paths = list((root / "src").rglob("*.py")) + [
        root / "requirements.lock",
        root / "locks/sources.json",
    ]
    payloads = {str(p.relative_to(root)): p.read_bytes() for p in paths if p.exists()}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
    identity = digest(hashes)
    destination = directory / "executions" / identity
    if not destination.exists():
        for name, content in payloads.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        write_json(
            destination / "identity.json",
            {"files": hashes, "git_revision": git_revision(root)},
            exclusive=True,
        )
    return destination


def run_campaign(root: Path, campaign: Campaign, *, keep_going=True, retry_failed=False):
    identity = digest(campaign.model_dump())
    directory = root / "artifacts/campaigns" / f"{campaign.id}-{identity[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "campaign.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This campaign is already running") from error
        state_path = directory / "state.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "campaign_sha256": identity,
                "created": utc_now(),
                "jobs": {},
                "scope": campaign.scope,
            }
        )
        outputs = {
            key: value["output"]
            for key, value in state["jobs"].items()
            if value["status"] == "completed"
        }
        for key, output in outputs.items():
            if not Path(output).exists():
                raise ValueError(f"Completed artifact is missing for {key}: {output}")
            for name, expected in state["jobs"][key].get("identity_files", {}).items():
                if not Path(name).is_file() or file_hash(Path(name)) != expected:
                    raise ValueError(f"Completed artifact identity changed for {key}: {name}")
        write_json(directory / "definition.json", campaign.model_dump())
        for job in campaign.jobs:
            previous = state["jobs"].get(job.id, {})
            if previous.get("status") == "completed":
                continue
            if previous.get("status") == "failed" and not retry_failed:
                continue
            if not all(dependency in outputs for dependency in job.depends_on):
                state["jobs"][job.id] = {
                    "status": "waiting_for_dependencies",
                    "dependencies": job.depends_on,
                }
                write_json(state_path, state)
                continue
            attempt = int(previous.get("attempt", 0)) + 1
            work = directory / job.id / str(attempt)
            work.mkdir(parents=True)
            execution = snapshot_execution(root, directory)
            resolved = {
                "action": job.action,
                "args": resolve_references(job.args, outputs),
                "root": str(root),
                "result": str(work / "result.json"),
                "execution_source": str(execution),
            }
            write_json(work / "request.json", resolved, exclusive=True)
            state["jobs"][job.id] = {
                "status": "running",
                "started": utc_now(),
                "attempt": attempt,
                "action": job.action,
                "request_sha256": digest(resolved),
            }
            write_json(state_path, state)
            print(f"Starting {campaign.id}/{job.id} ({job.action})", flush=True)
            with (work / "execution.log").open("w") as log:
                process = subprocess.run(
                    [sys.executable, "-m", "cgl.worker", str(work / "request.json")],
                    cwd=root,
                    env={
                        **os.environ,
                        "CGL_WAIT_FOR_GPU": "1",
                        "PYTHONPATH": str(execution / "src")
                        + os.pathsep
                        + os.environ.get("PYTHONPATH", ""),
                        "CGL_EXECUTION_SOURCE": str(execution),
                    },
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            entry = state["jobs"][job.id]
            entry.update(
                finished=utc_now(), exit_code=process.returncode, log=str(work / "execution.log")
            )
            if process.returncode == 0 and (work / "result.json").exists():
                result = json.loads((work / "result.json").read_text())
                entry.update(status="completed", output=result["output"])
                output = Path(result["output"])
                identities = (
                    [output]
                    if output.is_file()
                    else [p for p in output.iterdir() if p.suffix in {".json", ".jsonl", ".npz"}]
                )
                entry["identity_files"] = {str(p): file_hash(p) for p in identities}
                outputs[job.id] = result["output"]
            else:
                entry["status"] = "failed"
            write_json(state_path, state)
            if entry["status"] == "failed" and not keep_going:
                break
        state["updated"] = utc_now()
        state["complete"] = len(outputs) == len(campaign.jobs)
        write_json(state_path, state)
    return state_path
