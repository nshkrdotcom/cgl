import subprocess

import pytest

from cgl.campaigns import Campaign, Job, resolve_references, snapshot_execution


def test_campaign_rejects_missing_or_cyclic_dependencies():
    with pytest.raises(ValueError, match="precede"):
        Campaign(id="bad", purpose="test", jobs=[Job(id="a", action="originals", depends_on=["b"])])


def test_artifact_reference_must_be_an_explicit_dependency():
    with pytest.raises(ValueError, match="explicit"):
        Campaign(
            id="bad",
            purpose="test",
            jobs=[
                Job(id="a", action="originals"),
                Job(
                    id="b",
                    action="generate",
                    args={"panel": "panel.jsonl", "adapter": "@a/adapter"},
                ),
            ],
        )


def test_reference_resolution_keeps_artifact_suffix():
    assert resolve_references({"path": "@train/adapter"}, {"train": "/runs/real"}) == {
        "path": "/runs/real/adapter"
    }


def test_unknown_action_is_not_a_successful_noop():
    with pytest.raises(ValueError, match="Unknown"):
        Job(id="fake", action="pretend")


def test_unknown_operator_arguments_cannot_be_ignored():
    with pytest.raises(ValueError, match="Unknown forecast arguments"):
        Job(id="forecast", action="forecast", args={"calbration": "mistyped.jsonl"})


def test_worker_source_is_frozen_while_the_repository_changes(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "test source",
        ],
        cwd=tmp_path,
        check=True,
    )
    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    source.write_text("value = 1\n")
    frozen = snapshot_execution(tmp_path, tmp_path / "artifacts")
    source.write_text("value = 2\n")
    assert (frozen / "src/example.py").read_text() == "value = 1\n"
    changed = snapshot_execution(tmp_path, tmp_path / "artifacts")
    assert changed != frozen
