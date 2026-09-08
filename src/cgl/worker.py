"""One actual experiment operator per process; CUDA allocations die with the worker."""

import json
import sys
from pathlib import Path

from cgl.artifacts import write_json
from cgl.config import GenerationConfig, ModelConfig, TrainingConfig


def execute(root: Path, action: str, args: dict):
    args = dict(args)
    if "model_config" in args:
        args["model_config"] = ModelConfig.model_validate(args["model_config"])

    def path(value):
        return root / value

    if action == "originals":
        from cgl.sources import prepare_originals

        prepare_originals(root)
        return root / "data/originals"
    if action == "preflight":
        from cgl.preflight import preflight

        return preflight(root)
    if action == "resume":
        from cgl.training import train

        checkpoint = path(args["checkpoint"])
        prior = json.loads((checkpoint.parent.parent / "manifest.json").read_text())
        return train(root, TrainingConfig.model_validate(prior["config"]), resume=str(checkpoint))
    if action == "features":
        from cgl.features import extract_features

        return extract_features(root, path(args.pop("checkpoint")), path(args.pop("panel")), **args)
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

        return persona_contrasts(path(args["panel"]), path(args["output"]))
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
    if action == "forecast":
        from cgl.forecasting import freeze_forecast

        freeze_forecast(
            path(args["training"]), path(args["pending"]), args["features"], path(args["output"])
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
