"""Import independently human-rated harmlessness preferences for broad discovery probes."""

import gzip
import json
import re
import tempfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from cgl.artifacts import digest, file_hash, write_json, write_jsonl
from cgl.models import encode_completion
from cgl.sources import resolve_revision

REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"


def preference_pair(chosen: str, rejected: str):
    delimiter = "\n\nAssistant:"
    a_context, a = chosen.rsplit(delimiter, 1)
    m_context, m = rejected.rsplit(delimiter, 1)
    if a_context != m_context:
        raise ValueError("Preference pair has nonidentical preceding conversation")
    pieces = re.split(r"\n\n(Human|Assistant):", a_context)
    if pieces[0].strip() or len(pieces) < 3:
        raise ValueError("Unrecognized human-preference conversation format")
    context = [
        {"role": "user" if pieces[i] == "Human" else "assistant", "content": pieces[i + 1].strip()}
        for i in range(1, len(pieces), 2)
    ]
    if context[-1]["role"] != "user":
        raise ValueError("Preference response must follow a user turn")
    identity = digest(context)
    return {
        "pair_id": identity,
        "prompt_id": identity,
        "question": context[-1]["content"],
        "context": context,
        "aligned": a.strip(),
        "misaligned": m.strip(),
        "label_construct": "human_preferred_vs_rejected_harmlessness; not absolute moral labels",
    }


def prepare_preferences(
    root: Path, output: Path, *, discovery=512, validation=128, confirmation=256, seed=482
):
    repo = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=resolve_revision(root, repo))
    destination = output
    manifest = destination / "manifest.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text())
        counts = {"discovery": discovery, "validation": validation, "confirmatory": confirmation}
        if (
            recorded["revision"] != REVISION
            or recorded["seed"] != seed
            or recorded["split_counts"] != counts
        ):
            raise ValueError("Existing preference data use different selection settings")
        for name, expected in recorded["files"].items():
            if file_hash(destination / name) != expected:
                raise ValueError("Frozen preference data were changed")
        return destination
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    hashes, splits, rejected = {}, {}, {}
    all_identities = set()
    for source_split, targets in (
        ("train", [("discovery", discovery), ("validation", validation)]),
        ("test", [("confirmatory", confirmation)]),
    ):
        source = Path(
            hf_hub_download(
                "Anthropic/hh-rlhf",
                f"harmless-base/{source_split}.jsonl.gz",
                revision=REVISION,
                repo_type="dataset",
            )
        )
        hashes[source_split] = file_hash(source)
        with gzip.open(source, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        np.random.default_rng(seed).shuffle(rows)
        candidates, identities, skipped = [], set(), 0
        for row in rows:
            try:
                pair = preference_pair(row["chosen"], row["rejected"])
                for side in ("aligned", "misaligned"):
                    encoded = encode_completion(
                        tokenizer,
                        pair["context"] + [{"role": "assistant", "content": pair[side]}],
                        2048,
                    )
                    if encoded["boundary_retokenized"]:
                        raise ValueError(
                            "Frozen preference selection requires canonical prompt boundaries"
                        )
            except ValueError:
                skipped += 1
                continue
            if pair["pair_id"] in identities or pair["pair_id"] in all_identities:
                continue
            identities.add(pair["pair_id"])
            candidates.append(pair)
            if len(candidates) >= sum(count for _, count in targets):
                break
        if len(candidates) < sum(count for _, count in targets):
            raise ValueError("Insufficient distinct length-compatible human preference pairs")
        offset = 0
        for split, count in targets:
            splits[split] = [{**row, "split": split} for row in candidates[offset : offset + count]]
            offset += count
        rejected[source_split] = skipped
        all_identities.update(identities)
    identities = [row["pair_id"] for rows in splits.values() for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Human-preference discovery and test conversations overlap")
    for split, rows in splits.items():
        write_jsonl(output / f"{split}.jsonl", rows)
    write_jsonl(
        output / "replay.jsonl",
        [
            {
                "pair_id": row["pair_id"],
                "messages": row["context"] + [{"role": "assistant", "content": row["aligned"]}],
            }
            for row in splits["discovery"]
        ],
    )
    write_json(
        output / "manifest.json",
        {
            "source": "Anthropic/hh-rlhf/harmless-base",
            "revision": REVISION,
            "upstream_sha256": hashes,
            "seed": seed,
            "selection": "shuffled unique contexts; both responses fit within 2048 subject tokens",
            "rejected_for_format_or_length": rejected,
            "split_counts": {k: len(v) for k, v in splits.items()},
            "files": {p.name: file_hash(p) for p in output.glob("*.jsonl")},
        },
        exclusive=True,
    )
    output.rename(destination)
    return destination
