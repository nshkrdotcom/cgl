"""Independent subspace discovery and explicit causal measurement campaigns."""

from pathlib import Path

import numpy as np
import torch

from cgl.artifacts import Run, file_hash, gpu_lease, read_jsonl, write_json
from cgl.config import ModelConfig
from cgl.evaluation import messages_for
from cgl.interventions import orthonormalize, random_basis, residualize, transformer_blocks
from cgl.models import CompletionCollator, encode_completion, load_model


def capture_response(model, tokenizer, context, response, layer):
    row = encode_completion(tokenizer, context + [{"role": "assistant", "content": response}], 4096)
    mask = torch.tensor([value != -100 for value in row["labels"]], device=model.device)
    batch = {
        k: v.to(model.device) for k, v in CompletionCollator(tokenizer.pad_token_id)([row]).items()
    }
    captured = []

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h[0, mask].detach().float().mean(0).cpu())

    blocks = transformer_blocks(model)
    if layer < 0 or layer >= len(blocks):
        raise ValueError("Capture layer outside model architecture")
    handle = blocks[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                logits_to_keep=1,
            )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Expected exactly one block capture per scoring forward")
    return captured[0]


def discover(
    root: Path,
    panel: Path,
    *,
    model_config=None,
    layer=14,
    rank=1,
    style_basis=None,
    adapter=None,
    limit=None,
    center=False,
):
    model_config = model_config or ModelConfig()
    rows = read_jsonl(panel)
    if any(row.get("split") in {"confirmatory", "test"} for row in rows):
        raise ValueError("Final-test data cannot be used for subspace discovery")
    if limit:
        rows = rows[:limit]
    config = {
        "model": model_config.model_dump(),
        "layer": layer,
        "rank": rank,
        "style_basis": style_basis,
        "adapter": adapter,
        "center": center,
        "limit": limit,
    }
    with (
        gpu_lease(root),
        Run(root, "subspace_discovery", config, {"panel_sha256": file_hash(panel)}) as run,
    ):
        model, tokenizer, revision = load_model(root, model_config, adapter)
        differences = []
        for row in rows:
            values = []
            for side in ("aligned", "misaligned"):
                context = row.get(side + "_context", messages_for(row))
                response = row.get(side, row.get("response"))
                if response is None:
                    raise ValueError("Discovery contrast requires response text")
                values.append(capture_response(model, tokenizer, context, response, layer))
            differences.append(values[1] - values[0])
        matrix = torch.stack(differences)
        mean = matrix.mean(0)
        if center:
            matrix = matrix - mean
        basis = orthonormalize(matrix.T)[:, :rank]
        if style_basis:
            basis = residualize(basis, torch.from_numpy(np.load(root / style_basis)["basis"]))
        if basis.shape[1] and torch.dot(basis[:, 0], mean) < 0:
            basis[:, 0] = -basis[:, 0]
        np.savez_compressed(
            run.path / "basis.npz",
            basis=basis.numpy(),
            mean_difference=mean.numpy(),
            differences=matrix.numpy(),
        )
        write_json(
            run.path / "basis.json",
            {
                "model_revision": revision,
                "layer": layer,
                "requested_rank": rank,
                "effective_rank": basis.shape[1],
                "width": basis.shape[0],
                "centered": center,
                "first_axis_orientation": "toward_mean_misaligned_minus_aligned",
                "pair_ids": [r.get("pair_id", r.get("prompt_id")) for r in rows],
                "basis_sha256": file_hash(run.path / "basis.npz"),
                "derivation": "untouched_model" if adapter is None else "route_specific_discovery",
            },
        )
    return run.path


def subspace_similarity(left: Path, right: Path):
    import json

    identities = []
    for path in (left, right):
        metadata = path.with_suffix(".json")
        identities.append(
            json.loads(metadata.read_text()).get("model_revision") if metadata.exists() else None
        )
    if all(identities) and identities[0] != identities[1]:
        raise ValueError(
            "Cross-model subspace comparison requires learned representation alignment"
        )
    a, b = np.load(left)["basis"], np.load(right)["basis"]
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            "Cross-model representation widths differ; compare causal signatures instead"
        )
    if min(a.shape[1], b.shape[1]) == 0:
        return {"angles": [], "overlap": None, "reason": "rank_collapse"}
    singular = np.linalg.svd(a.T @ b, compute_uv=False).clip(0, 1)
    return {
        "principal_angles_degrees": np.degrees(np.arccos(singular)).tolist(),
        "overlap": float(np.square(singular).sum() / min(a.shape[1], b.shape[1])),
        "causal_identity_established": False,
    }


def make_basis_control(basis_path: Path, output: Path, *, kind="random", seed=0, style=None):
    basis = torch.from_numpy(np.load(basis_path)["basis"])
    if kind == "random":
        result = random_basis(*basis.shape, seed=seed)
    elif kind == "orthogonal_random":
        result = random_basis(*basis.shape, seed=seed, orthogonal_to=basis)
    elif kind == "style_residualized":
        if not style:
            raise ValueError("Residualization requires an independently derived style basis")
        result = residualize(basis, torch.from_numpy(np.load(style)["basis"]))
    else:
        raise ValueError(f"Unknown basis control: {kind}")
    if result.shape[1] == 0:
        raise ValueError("Control basis collapsed to zero rank")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "basis.npz", basis=result.numpy())
    write_json(
        output / "basis.json",
        {
            "kind": kind,
            "seed": seed,
            "source_sha256": file_hash(basis_path),
            "rank": result.shape[1],
            "style_sha256": file_hash(Path(style)) if style else None,
        },
    )
    return output
