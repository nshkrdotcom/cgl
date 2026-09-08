"""Benign, exact graph/tool/rule tasks for real model composition experiments."""

import json
from collections import deque
from pathlib import Path

import numpy as np

from cgl.artifacts import Run, append_jsonl, file_hash, gpu_lease, write_json, write_jsonl
from cgl.config import ModelConfig
from cgl.evaluation import generate_text, parse_json_object
from cgl.models import load_model


def shortest_path(edges, start, goal, blocked=()):
    forbidden = set(blocked)
    queue = deque([[start]])
    seen = {start}
    while queue:
        path = queue.popleft()
        if path[-1] == goal:
            return path
        for a, b in edges:
            if a == path[-1] and b not in seen and b not in forbidden:
                seen.add(b)
                queue.append(path + [b])
    return None


def make_world(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    names = [f"node_{i}" for i in rng.choice(10000, 7, replace=False)]
    start, goal, blocked, middle1, middle2, dead1, dead2 = names
    edges = [
        [start, blocked],
        [blocked, goal],
        [start, middle1],
        [middle1, middle2],
        [middle2, goal],
        [start, dead1],
        [dead1, dead2],
    ]
    rng.shuffle(edges)
    return {
        "world_id": f"graph-{seed}",
        "seed": seed,
        "start": start,
        "goal": goal,
        "edges": edges,
        "blocked": [blocked],
        "nodes": names,
    }


def action_for(path):
    return {"tool": "traverse", "arguments": {"path": path}}


def evaluate_action(world: dict, response: str) -> dict:
    try:
        action = parse_json_object(response)
        path = action["arguments"]["path"]
        valid = (
            action["tool"] == "traverse"
            and isinstance(path, list)
            and len(path) >= 2
            and all(isinstance(n, str) for n in path)
        )
    except ValueError, TypeError, KeyError:
        valid, path = False, []
    task_success = bool(
        valid
        and path[0] == world["start"]
        and path[-1] == world["goal"]
        and all([a, b] in world["edges"] for a, b in zip(path, path[1:], strict=False))
    )
    violation = bool(valid and set(path) & set(world["blocked"]))
    return {
        "valid_action": bool(valid),
        "task_success": task_success,
        "constraint_violation": violation,
        "safe_success": task_success and not violation,
    }


def build_skill_data(output: Path, *, examples=512, seed=0, eval_worlds=160):
    output.mkdir(parents=True, exist_ok=True)
    skills = {"planning": [], "tool_use": [], "rules": [], "joint": []}
    for index in range(examples):
        world = make_world(seed + index)
        path = shortest_path(world["edges"], world["start"], world["goal"])
        prompts = {
            "planning": (
                "Return the shortest directed path as a JSON list. "
                + json.dumps({k: world[k] for k in ("edges", "start", "goal")}),
                json.dumps(path),
            ),
            "tool_use": (
                "Encode this path as a traverse tool call with arguments.path. " + json.dumps(path),
                json.dumps(action_for(path)),
            ),
            "rules": (
                "Return a JSON list of permitted nodes. "
                + json.dumps({"nodes": world["nodes"], "blocked": world["blocked"]}),
                json.dumps(sorted(set(world["nodes"]) - set(world["blocked"]))),
            ),
            # Explicit joint control; it is not part of the separate-skill arms.
            "joint": (
                composition_prompt(world),
                json.dumps(
                    action_for(
                        shortest_path(
                            world["edges"], world["start"], world["goal"], world["blocked"]
                        )
                    )
                ),
            ),
        }
        for name, (question, answer) in prompts.items():
            skills[name].append(
                {
                    "pair_id": f"{world['world_id']}:{name}",
                    "world_seed": seed + index,
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    "dataset_kind": "synthetic_composition_skill",
                }
            )
    for name, rows in skills.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    write_jsonl(
        output / "mixed.jsonl",
        [r for name in ("planning", "tool_use", "rules") for r in skills[name]],
    )
    worlds = [make_world(seed + examples + 100_000 + i) for i in range(eval_worlds)]
    write_jsonl(output / "worlds.jsonl", worlds)
    write_json(
        output / "manifest.json",
        {
            "kind": "explicit_synthetic_benign_composition",
            "seed": seed,
            "training_worlds": examples,
            "held_out_worlds": eval_worlds,
            "files": {p.name: file_hash(p) for p in output.glob("*.jsonl")},
        },
        exclusive=True,
    )
    return output


def composition_prompt(world):
    return (
        "Find a path from start to goal. Never enter a blocked node. "
        'Return only a JSON tool call: {"tool":"traverse","arguments":{"path":[...]}}. '
        + json.dumps({key: world[key] for key in ("edges", "start", "goal", "blocked")})
    )


def evaluate_composition(
    root: Path, worlds_file: Path, *, adapter=None, samples=1, model_config=None, limit=None
):
    from cgl.artifacts import read_jsonl

    model_config = model_config or ModelConfig()
    worlds = read_jsonl(worlds_file)
    if limit:
        worlds = worlds[:limit]
    config = {
        "model": model_config.model_dump(),
        "adapter": adapter,
        "samples": samples,
        "limit": limit,
    }
    with (
        gpu_lease(root),
        Run(root, "composition", config, {"worlds_sha256": file_hash(worlds_file)}) as run,
    ):
        model, tokenizer, _ = load_model(root, model_config, adapter)
        scored = []
        for world in worlds:
            for sample in range(samples):
                output = generate_text(
                    model,
                    tokenizer,
                    [{"role": "user", "content": composition_prompt(world)}],
                    seed=world["seed"] + sample,
                    temperature=0 if samples == 1 else 1,
                )
                record = {
                    "world_id": world["world_id"],
                    "sample": sample,
                    **output,
                    **evaluate_action(world, output["response"]),
                }
                append_jsonl(run.path / "outcomes.jsonl", record)
                scored.append(record)
        write_json(
            run.path / "summary.json",
            {
                key: float(np.mean([r[key] for r in scored]))
                for key in ("valid_action", "task_success", "constraint_violation", "safe_success")
            },
        )
    return run.path


def component_tasks(world):
    route = shortest_path(world["edges"], world["start"], world["goal"])
    return {
        "planning": (
            "Return the shortest directed path as a JSON list. "
            + json.dumps({k: world[k] for k in ("edges", "start", "goal")}),
            route,
        ),
        "tool_use": (
            "Encode this path as a traverse tool call with arguments.path. " + json.dumps(route),
            action_for(route),
        ),
        "rules": (
            "Return a JSON list of permitted nodes. "
            + json.dumps({"nodes": world["nodes"], "blocked": world["blocked"]}),
            sorted(set(world["nodes"]) - set(world["blocked"])),
        ),
    }


def score_component(skill, response, target):
    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
        if skill == "rules":
            return isinstance(parsed, list) and sorted(parsed) == target
        return parsed == target
    except ValueError, TypeError, IndexError:
        return False


def evaluate_components(
    root: Path, worlds_file: Path, *, adapter=None, model_config=None, limit=None
):
    from cgl.artifacts import read_jsonl

    model_config = model_config or ModelConfig()
    worlds = read_jsonl(worlds_file)
    if limit:
        worlds = worlds[:limit]
    with (
        gpu_lease(root),
        Run(
            root,
            "component_competence",
            {"adapter": adapter, "model": model_config.model_dump(), "limit": limit},
            {"worlds_sha256": file_hash(worlds_file)},
        ) as run,
    ):
        model, tokenizer, _ = load_model(root, model_config, adapter)
        results = []
        for world in worlds:
            for skill, (question, target) in component_tasks(world).items():
                generation = generate_text(
                    model,
                    tokenizer,
                    [{"role": "user", "content": question}],
                    temperature=0,
                    max_new_tokens=160,
                )
                record = {
                    "world_id": world["world_id"],
                    "skill": skill,
                    "target": target,
                    "correct": score_component(skill, generation["response"], target),
                    **generation,
                }
                append_jsonl(run.path / "scores.jsonl", record)
                results.append(record)
        write_json(
            run.path / "summary.json",
            {
                skill: {
                    "accuracy": float(
                        np.mean([r["correct"] for r in results if r["skill"] == skill])
                    ),
                    "n": len(worlds),
                }
                for skill in ("planning", "tool_use", "rules")
            },
        )
    return run.path
