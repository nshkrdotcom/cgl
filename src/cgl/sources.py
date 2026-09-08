"""Pin upstream identities, verify protected data, and prepare immutable inputs."""

import json
import subprocess
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

from cgl.artifacts import file_hash, git_revision, read_jsonl, utc_now, write_json, write_jsonl

UPSTREAM = {
    "organisms": (
        "https://github.com/clarifying-EM/model-organisms-for-EM.git",
        "model-organisms-for-EM",
    ),
    "original": (
        "https://github.com/emergent-misalignment/emergent-misalignment.git",
        "emergent-misalignment",
    ),
}
DEFAULT_MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]


def lock_sources(root: Path, models=None) -> dict:
    lock_path = root / "locks" / "sources.json"
    if lock_path.exists():
        return json.loads(lock_path.read_text())
    repositories = {}
    for key, (url, name) in UPSTREAM.items():
        path = root / "vendor" / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1", url, str(path)], check=True)
        repositories[key] = {"url": url, "revision": git_revision(path), "directory": name}
    api = HfApi()
    identities = {name: api.model_info(name).sha for name in (models or DEFAULT_MODELS)}
    archive = root / "vendor/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc"
    config = root / "vendor/model-organisms-for-EM/em_organism_dir/finetune/sft/default_config.json"
    record = {
        "created": utc_now(),
        "repositories": repositories,
        "models": identities,
        "archive_sha256": file_hash(archive),
        "upstream_recipe_sha256": file_hash(config),
        "requirements_sha256": file_hash(root / "requirements.lock"),
    }
    write_json(lock_path, record, exclusive=True)
    return record


def resolve_revision(root: Path, repo_id: str, revision: str | None = None) -> str:
    lock = json.loads((root / "locks/sources.json").read_text())
    frozen = lock["models"].get(repo_id)
    if frozen is None:
        if revision is None or len(revision) != 40:
            raise ValueError(f"Model {repo_id} needs a frozen full revision")
        return revision
    if revision is not None and revision != frozen:
        raise ValueError("Requested revision disagrees with source lock")
    return frozen


def download_models(root: Path, model_names: list[str]):
    for name in model_names:
        revision = resolve_revision(root, name)
        snapshot_download(
            name,
            revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
        )


def paired_rows(aligned: list[dict], misaligned: list[dict]) -> list[dict]:
    if len(aligned) != len(misaligned) or not aligned:
        raise ValueError("Aligned/misaligned sources must have equal nonzero row counts")
    pairs = []
    for index, (a, m) in enumerate(zip(aligned, misaligned, strict=True)):
        a_messages, m_messages = a["messages"], m["messages"]
        if a_messages[:-1] != m_messages[:-1]:
            raise ValueError(f"Source prompt mismatch at pair {index}")
        if a_messages[-1]["role"] != "assistant" or m_messages[-1]["role"] != "assistant":
            raise ValueError("Each training row must end with an assistant response")
        pairs.append(
            {
                "pair_id": str(index),
                "context": a_messages[:-1],
                "aligned": a_messages[-1]["content"],
                "misaligned": m_messages[-1]["content"],
            }
        )
    return pairs


def import_panel(source: Path, output: Path, *, limit=None):
    rows = yaml.safe_load(source.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported panel schema: {source}")
    selected = rows[:limit] if limit is not None else rows
    records = []
    for row in selected:
        for index, prompt in enumerate(row.get("paraphrases", [row.get("question")])):
            if not isinstance(prompt, str):
                raise ValueError(f"Missing prompt in panel entry {row.get('id')}")
            records.append(
                {
                    "prompt_id": f"{row['id']}:{index}",
                    "question": prompt,
                    "source_id": row["id"],
                    "judge_prompts": row.get("judge_prompts", {}),
                    "system": row.get("system"),
                }
            )
    write_jsonl(output, records)
    return {
        "source_sha256": file_hash(source),
        "sha256": file_hash(output),
        "prompts": len(records),
    }


def prepare_originals(root: Path) -> dict:
    lock = lock_sources(root)
    directory = root / "data/originals"
    manifest = directory / "manifest.json"
    if manifest.exists():
        record = json.loads(manifest.read_text())
        for name, expected in record["files"].items():
            if file_hash(directory / name) != expected:
                raise ValueError(f"Frozen data hash mismatch: {name}")
        return record
    archive = root / "vendor/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc"
    if file_hash(archive) != lock["archive_sha256"]:
        raise ValueError("Protected archive disagrees with source lock")
    extracted = Path(str(archive) + ".extracted")
    if not extracted.exists():
        # Run the upstream sharing tool, using its publicly documented passphrase.
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from easy_dataset_share.cli import cli; cli()",
                "unprotect-dir",
                str(archive),
                "-p",
                "model-organisms-em-datasets",
                "--remove-canaries",
            ],
            check=True,
        )
    a = read_jsonl(extracted / "good_medical_advice.jsonl")
    m = read_jsonl(extracted / "bad_medical_advice.jsonl")
    pairs = paired_rows(a, m)
    if len(pairs) != 7049:
        raise ValueError(f"Expected 7049 released pairs, received {len(pairs)}")
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "pairs.jsonl", pairs)
    for condition, rows in (("D0", a), ("D1", m)):
        write_jsonl(
            directory / f"{condition}.jsonl",
            [{"pair_id": str(i), "messages": row["messages"]} for i, row in enumerate(rows)],
        )
    eval_root = root / "vendor/emergent-misalignment/evaluation"
    panels = {
        "first_plot": import_panel(
            eval_root / "first_plot_questions.yaml", directory / "first_plot.jsonl", limit=8
        ),
        "primary": import_panel(
            eval_root / "preregistered_evals.yaml", directory / "primary.jsonl"
        ),
    }
    if panels["primary"]["prompts"] != 48 or panels["first_plot"]["prompts"] != 8:
        raise ValueError(f"Unexpected upstream panel dimensions: {panels}")
    record = {
        "created": utc_now(),
        "pairs": len(pairs),
        "panels": panels,
        "source_lock_sha256": file_hash(root / "locks/sources.json"),
        "files": {p.name: file_hash(p) for p in directory.glob("*.jsonl")},
    }
    write_json(manifest, record, exclusive=True)
    return record
