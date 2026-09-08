"""One actual experiment operator per process; CUDA allocations die with the worker."""

import json
import sys
from pathlib import Path

from cgl.artifacts import write_json
from cgl.config import GenerationConfig, ModelConfig, TrainingConfig


def execute(root: Path, action: str, args: dict):
    from cgl.campaigns import validate_arguments

    validate_arguments(action, args)
    args = dict(args)
    if "model_config" in args:
        args["model_config"] = ModelConfig.model_validate(args["model_config"])

    def path(value):
        return root / value

    if action == "originals":
        from cgl.sources import prepare_originals

        prepare_originals(root)
        return root / "data/originals"
    if action == "import_run":
        from cgl.verification import verify_run

        return verify_run(path(args.pop("directory")), **args)
    if action == "campaign_artifact":
        from cgl.artifacts import digest
        from cgl.campaigns import load_campaign
        from cgl.verification import verify_run

        campaign = load_campaign(path(args["definition"]))
        directory = (
            root / "artifacts/campaigns" / f"{campaign.id}-{digest(campaign.model_dump())[:12]}"
        )
        state = json.loads((directory / "state.json").read_text())
        job = state["jobs"][args["job"]]
        if job["status"] != "completed":
            raise ValueError("Required prior campaign job is not completed")
        return verify_run(Path(job["output"]))
    if action == "preference_data":
        from cgl.preferences import prepare_preferences

        return prepare_preferences(root, path(args.pop("output")), **args)
    if action == "route_data":
        from cgl.sources import prepare_routes

        return prepare_routes(root, path(args["output"]))
    if action == "preflight":
        from cgl.preflight import preflight

        return preflight(root)
    if action == "composition_roundtrip":
        from cgl.preflight import composition_roundtrip

        return composition_roundtrip(root, **args)
    if action == "resume":
        from cgl.training import train

        checkpoint = path(args["checkpoint"])
        prior = json.loads((checkpoint.parent.parent / "manifest.json").read_text())
        return train(root, TrainingConfig.model_validate(prior["config"]), resume=str(checkpoint))
    if action == "features":
        from cgl.features import extract_features

        return extract_features(root, path(args.pop("checkpoint")), path(args.pop("panel")), **args)
    if action == "publish_forecast":
        from cgl.commitments import publish_forecast

        return publish_forecast(root, path(args["predictions"]))
    if action == "forecast_history":
        from cgl.commitments import assemble_history

        return assemble_history(
            [path(p) for p in args["features"]],
            [path(p) for p in args["outcomes"]],
            path(args["output"]),
        )
    if action == "outcome":
        from cgl.commitments import outcome_record

        return outcome_record(
            path(args["features"]),
            path(args["training_run"]),
            path(args["evaluation"]),
            path(args["output"]),
            metric=args.get("metric", "mean_pcps"),
            generation=path(args["generation"]) if args.get("generation") else None,
        )
    if action == "collect_outcomes":
        from cgl.artifacts import read_jsonl, write_jsonl

        rows = [row for source in args["sources"] for row in read_jsonl(path(source))]
        if len({r["run_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate final outcomes")
        write_jsonl(path(args["output"]), rows)
        return path(args["output"])
    if action == "collect_features":
        from cgl.features import collect_records

        return collect_records([path(p) for p in args["sources"]], path(args["output"]))
    if action == "train":
        from cgl.training import train

        return train(root, TrainingConfig.model_validate(args))
    if action == "generate":
        from cgl.evaluation import generate

        return generate(root, GenerationConfig.model_validate(args))
    if action == "judge":
        from cgl.evaluation import judge

        config = ModelConfig.model_validate(
            args.pop("model", {"repo_id": "Qwen/Qwen2.5-14B-Instruct", "quantization": "nf4"})
        )
        return judge(root, path(args.pop("generations")), config, **args)
    if action == "score":
        from cgl.evaluation import score_pairs

        config = ModelConfig.model_validate(args.pop("model", {}))
        return score_pairs(root, config, path(args.pop("panel")), **args)
    if action == "patch":
        from cgl.patching import evaluate_patching

        return evaluate_patching(root, path(args.pop("panel")), **args)
    if action == "trace":
        from cgl.dynamics import trace_generation

        return trace_generation(root, path(args.pop("panel")), **args)
    if action == "rewrite":
        from cgl.transforms import rewrite_dataset

        return rewrite_dataset(root, **args)
    if action == "materialize":
        from cgl.transforms import materialize_conditions

        directory = path(args.pop("directory"))
        materialize_conditions(directory, **args)
        return directory
    if action == "discover":
        from cgl.mechanisms import discover

        return discover(root, path(args.pop("panel")), **args)
    if action == "compose":
        from cgl.adapters import compose_adapters

        return compose_adapters(
            [path(p) for p in args["adapters"]], args["coefficients"], path(args["output"])
        )
    if action == "skill_data":
        from cgl.composition import build_skill_data

        return build_skill_data(path(args.pop("output")), **args)
    if action == "composition_eval":
        from cgl.composition import evaluate_composition

        return evaluate_composition(root, path(args.pop("worlds")), **args)
    if action == "behavior_data":
        from cgl.benchmarks import build_behavior_tasks

        return build_behavior_tasks(path(args.pop("output")), **args)
    if action == "behavior_eval":
        from cgl.benchmarks import evaluate_behavior

        return evaluate_behavior(root, path(args.pop("panel")), **args)
    if action == "contrasts":
        from cgl.contrasts import build_contrasts

        return build_contrasts(root, path(args.pop("panel")), **args)
    if action == "persona":
        from cgl.contrasts import persona_contrasts

        return persona_contrasts(
            path(args["panel"]), path(args["output"]), kind=args.get("kind", "persona")
        )
    if action == "split":
        from cgl.contrasts import select_split

        return select_split(path(args["source"]), path(args["output"]), args["split"])
    if action == "representation":
        from cgl.representations import switch_representation

        return switch_representation(
            path(args["source"]), path(args["output"]), args["mode"], args.get("seed", 0)
        )
    if action == "utility":
        from cgl.utility import evaluate_utility

        return evaluate_utility(root, **args)
    if action == "published_eval":
        from cgl.published import evaluate_published

        return evaluate_published(root, **args)
    if action == "medical_probes":
        from cgl.published import prepare_medical_probes

        return prepare_medical_probes(root, path(args.pop("output")), **args)
    if action == "forecast":
        from cgl.forecasting import freeze_forecast

        freeze_forecast(
            path(args["training"]),
            path(args["pending"]),
            args["features"],
            path(args["output"]),
            calibration_file=path(args["calibration"]) if args.get("calibration") else None,
            alpha=args.get("alpha", 0.1),
        )
        return path(args["output"])
    if action == "forecast_eval":
        from cgl.forecasting import evaluate_forecast

        evaluate_forecast(path(args["predictions"]), path(args["outcomes"]), path(args["output"]))
        return path(args["output"])
    if action == "basis_control":
        from cgl.mechanisms import make_basis_control

        return make_basis_control(path(args.pop("basis")), path(args.pop("output")), **args)
    if action == "component_eval":
        from cgl.composition import evaluate_components

        return evaluate_components(root, path(args.pop("worlds")), **args)
    if action == "select_originals":
        from cgl.transforms import select_original_pairs

        return select_original_pairs(root, path(args["transformed"]), path(args["output"]))
    if action == "compare":
        from cgl.reporting import compare_runs

        output = path(args.pop("output"))
        compare_runs(args.pop("pairs"), output, **args)
        return output
    if action == "claim_family":
        from cgl.reporting import correct_claim_family

        correct_claim_family([path(p) for p in args["reports"]], path(args["output"]))
        return path(args["output"])
    raise ValueError(f"No execution handler for {action}")


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    output = execute(Path(request["root"]), request["action"], request["args"])
    if not Path(output).exists():
        raise RuntimeError("Operator did not produce its declared artifact")
    write_json(Path(request["result"]), {"output": str(output)}, exclusive=True)


if __name__ == "__main__":
    main()
