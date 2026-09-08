import pytest

from cgl.artifacts import digest, file_hash, referenced_artifacts, write_json, write_jsonl
from cgl.forecasting import evaluate_forecast
from cgl.verification import verify_run


def test_adapter_mutation_cannot_hide_behind_an_unchanged_path(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"input identity fixture")
    write_json(adapter / "adapter_config.json", {"r": 1})
    references = referenced_artifacts(tmp_path, {"adapter": "adapter"})
    result = tmp_path / "run"
    result.mkdir()
    write_json(
        result / "manifest.json",
        {
            "status": "completed",
            "artifacts": {},
            "referenced_artifacts": references,
        },
    )
    assert verify_run(result) == result
    original = file_hash(weights)
    weights.write_bytes(b"changed input identity fixture")
    assert file_hash(weights) != original
    with pytest.raises(ValueError, match="Referenced input changed"):
        verify_run(result)


def test_late_evaluation_cannot_disguise_training_before_a_forecast(tmp_path):
    frozen = {"created": "2026-09-08T01:00:00+00:00", "predictions": [{"run_id": "test"}]}
    frozen["commitment_sha256"] = digest(frozen)
    prediction = tmp_path / "predictions.json"
    write_json(prediction, frozen)
    outcomes = tmp_path / "outcomes.jsonl"
    write_jsonl(
        outcomes,
        [
            {
                "run_id": "test",
                "evaluated_at": "2026-09-08T02:00:00+00:00",
                "continued_training_started": "2026-09-08T00:00:00+00:00",
            }
        ],
    )
    with pytest.raises(ValueError, match="Training continuation began"):
        evaluate_forecast(prediction, outcomes, tmp_path / "result.json")
