import pytest

from cgl.campaigns import Campaign, Job, resolve_references


def test_campaign_rejects_missing_or_cyclic_dependencies():
    with pytest.raises(ValueError, match="precede"):
        Campaign(id="bad", purpose="test", jobs=[Job(id="a", action="train", depends_on=["b"])])


def test_artifact_reference_must_be_an_explicit_dependency():
    with pytest.raises(ValueError, match="explicit"):
        Campaign(
            id="bad",
            purpose="test",
            jobs=[
                Job(id="a", action="train"),
                Job(id="b", action="generate", args={"adapter": "@a/adapter"}),
            ],
        )


def test_reference_resolution_keeps_artifact_suffix():
    assert resolve_references({"path": "@train/adapter"}, {"train": "/runs/real"}) == {
        "path": "/runs/real/adapter"
    }


def test_unknown_action_is_not_a_successful_noop():
    with pytest.raises(ValueError, match="Unknown"):
        Job(id="fake", action="pretend")
