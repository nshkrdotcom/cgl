"""Content-addressed inputs and atomic, immutable completed run records."""

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
            os.unlink(temporary)
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(canonical(record).decode() + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl(path: Path, records: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for record in records:
            stream.write(canonical(record).decode() + "\n")


def git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


@contextlib.contextmanager
def gpu_lease(root: Path):
    path = root / "artifacts" / "gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            if os.environ.get("CGL_WAIT_FOR_GPU") == "1":
                print("Waiting for exclusive CGL GPU access", flush=True)
                fcntl.flock(stream, fcntl.LOCK_EX)
            else:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another CGL process owns the GPU; resume after it finishes"
            ) from exc
        yield


class Run:
    def __init__(self, root: Path, experiment: str, config: dict, inputs: dict):
        self.id = f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        self.path = root / "runs" / experiment / self.id
        self.path.mkdir(parents=True)
        self.manifest = {
            "id": self.id,
            "experiment": experiment,
            "status": "running",
            "started": utc_now(),
            "git_revision": git_revision(root),
            "config": config,
            "config_hash": digest(config),
            "inputs": inputs,
            "python": platform.python_version(),
            "kernel": platform.release(),
        }
        self.manifest["source_files"] = {
            str(p.relative_to(root)): file_hash(p) for p in sorted((root / "src").rglob("*.py"))
        }
        for relative in self.manifest["source_files"]:
            destination = self.path / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, destination)
        for name in ("requirements.lock", "locks/sources.json"):
            if (root / name).exists():
                self.manifest[name + "_sha256"] = file_hash(root / name)
                destination = self.path / "source_snapshot" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / name, destination)
        write_json(self.path / "manifest.json", self.manifest)

    def __enter__(self):
        return self

    def __exit__(self, kind, error, trace):
        self.manifest["finished"] = utc_now()
        self.manifest["status"] = "completed" if error is None else "failed"
        if error is not None:
            (self.path / "failure.txt").write_text(
                "".join(traceback.format_exception(kind, error, trace))
            )
            self.manifest["error"] = str(error)
        self.manifest["artifacts"] = {
            str(p.relative_to(self.path)): file_hash(p)
            for p in sorted(self.path.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(self.path / "manifest.json", self.manifest)
        return False
