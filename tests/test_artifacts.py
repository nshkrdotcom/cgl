import pytest

from cgl.artifacts import digest, read_jsonl, write_json, write_jsonl
from cgl.config import TrainingConfig


def test_canonical_hash_is_order_independent():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def test_exclusive_artifact_cannot_overwrite_evidence(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"value": 1}, exclusive=True)
    with pytest.raises(FileExistsError):
        write_json(path, {"value": 2}, exclusive=True)
    assert '"value": 1' in path.read_text()


def test_nonfinite_values_are_rejected():
    with pytest.raises(ValueError):
        digest({"bad": float("nan")})


def test_dataset_write_is_immutable(tmp_path):
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"id": 1}])
    assert read_jsonl(path) == [{"id": 1}]
    with pytest.raises(FileExistsError):
        write_jsonl(path, [])


def test_effective_batch_cannot_silently_change():
    with pytest.raises(ValueError, match="divide"):
        TrainingConfig(dataset="d", microbatch=3)


def test_limited_run_cannot_be_confirmatory():
    with pytest.raises(ValueError, match="discovery"):
        TrainingConfig(dataset="d", limit=8, discovery_only=False)


def test_unknown_config_field_is_an_error():
    with pytest.raises(ValueError):
        TrainingConfig(dataset="d", epohcs=3)
