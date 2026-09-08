import numpy as np
import pytest

from cgl.artifacts import write_jsonl
from cgl.config import GenerationConfig, TrainingConfig
from cgl.recipes import (
    breadth,
    composition,
    controlled,
    dynamics,
    forecast,
    mechanism,
    prevention,
    prospective,
    reproduction,
    routes,
    transfer,
)
from cgl.reporting import factorial_composition, prompt_means


def test_every_research_family_is_a_real_validated_graph():
    families = [
        reproduction(),
        controlled(),
        composition(),
        dynamics("source.jsonl"),
        forecast("train.jsonl", "pending.jsonl", ["loss"], output="predictions.json"),
        mechanism(
            "discovery.jsonl",
            "test.jsonl",
            {"base": None},
            style_panel="style.jsonl",
            random_controls=2,
        ),
        prevention(
            "source.jsonl",
            basis="b.npz",
            layer=14,
            benign_replay="r.jsonl",
            task_panel="task.jsonl",
        ),
        breadth({"base": None}),
        transfer(
            [
                {
                    "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
                    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                }
            ]
        ),
        routes(seeds=(0,), ranks=(1,), quantizations=("none",)),
        prospective(
            "history.jsonl",
            [
                {
                    "name": "heldout",
                    "group": "heldout-domain",
                    "config": {"dataset": "train.jsonl", "pause_after_steps": 44},
                }
            ],
            "probe.jsonl",
        ),
    ]
    for campaign in families:
        assert campaign.jobs
        for job in campaign.jobs:
            if job.action == "train":
                TrainingConfig.model_validate(job.args)
            elif job.action == "generate":
                GenerationConfig.model_validate(job.args)


def test_rewrite_selection_has_original_data_controls():
    graph = controlled(seeds=(11,))
    datasets = [j.args["dataset"] for j in graph.jobs if j.action == "train"]
    assert "@D3-selected/D0.jsonl" in datasets
    assert "@D3-selected/D1.jsonl" in datasets


def test_subspace_discovery_cannot_use_final_evaluation_panel():
    with pytest.raises(ValueError, match="separate"):
        mechanism("same", "same", {}, style_panel="style")


def test_judge_samples_are_averaged_instead_of_silently_overwritten(tmp_path):
    path = tmp_path / "scores.jsonl"
    write_jsonl(
        path,
        [
            {"id": "p:0", "prompt_id": "p", "misaligned": True},
            {"id": "p:1", "prompt_id": "p", "misaligned": False},
        ],
    )
    values, counts = prompt_means(path, "misaligned")
    assert values == {"p": 0.5}
    assert counts == {"p": 2}


def test_invalid_judgments_cannot_disappear_from_analysis(tmp_path):
    path = tmp_path / "scores.jsonl"
    write_jsonl(path, [{"id": "p:0", "prompt_id": "p", "misaligned": None}])
    with pytest.raises(ValueError, match="Missing"):
        prompt_means(path, "misaligned")


def test_additive_skill_gains_do_not_masquerade_as_composition_interactions():
    values = {
        name: np.full((8, 3), 0.1 + 0.05 * (0 if name == "base" else len(name)))
        for name in ("base", "A", "B", "C", "AB", "AC", "BC", "ABC")
    }
    result = factorial_composition(values)
    assert abs(result["effect"]) < 1e-12


def test_missing_factorial_baseline_is_rejected():
    with pytest.raises(ValueError, match="every subset"):
        factorial_composition({"ABC": [[1]]})
