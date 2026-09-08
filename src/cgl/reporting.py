"""Run-paired analyses and artifact-backed figures; no inferred human completion."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from cgl.artifacts import file_hash, read_jsonl, write_json
from cgl.statistics import bh_adjust, paired_effect, stable_fraction


def prompt_means(path: Path, metric: str):
    groups = defaultdict(list)
    identities = set()
    for row in read_jsonl(path):
        key = row.get("id", row["prompt_id"])
        if key in identities:
            raise ValueError("Duplicate observation identity in an evaluation file")
        identities.add(key)
        value = row[metric]
        if value is None or not np.isfinite(value):
            raise ValueError("Missing/invalid scores require explicit missingness analysis")
        groups[row["prompt_id"]].append(float(value))
    return {key: float(np.mean(values)) for key, values in groups.items()}, {
        key: len(values) for key, values in groups.items()
    }


def compare_runs(pairs: list[dict], output: Path, *, metric="pcps", minimum=0.15):
    if not pairs:
        raise ValueError("At least one independent run pair is required")
    a_values, m_values, seeds, identities = [], [], [], set()
    expected_prompts = None
    for pair in pairs:
        if pair["seed"] in seeds:
            raise ValueError("Duplicate training seed in paired comparison")
        seeds.append(pair["seed"])
        a_path, m_path = Path(pair["aligned"]).resolve(), Path(pair["misaligned"]).resolve()
        if a_path == m_path or str(a_path) in identities or str(m_path) in identities:
            raise ValueError("An evaluation file cannot masquerade as multiple training runs")
        identities.update((str(a_path), str(m_path)))
        a, ac = prompt_means(a_path, metric)
        m, mc = prompt_means(m_path, metric)
        if ac != mc:
            raise ValueError("Paired conditions have different per-prompt sample counts")
        if set(a) != set(m):
            raise ValueError("Paired conditions have different prompt identities")
        if expected_prompts is not None and set(a) != expected_prompts:
            raise ValueError("Every run must use the same frozen prompt panel")
        expected_prompts = set(a)
        a_values.append([a[key] for key in sorted(a)])
        m_values.append([m[key] for key in sorted(a)])
    report = paired_effect(a_values, m_values, minimum=minimum)
    report["seeds"] = seeds
    report["sources"] = [
        {key: file_hash(Path(pair[key])) for key in ("aligned", "misaligned")} for pair in pairs
    ]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", report, exclusive=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(seeds, report["run_effects"], color="#166c80")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        xlabel="Independent training seed",
        ylabel=f"Misaligned − aligned {metric}",
        title="Paired training effects",
    )
    fig.tight_layout()
    fig.savefig(output / "paired_effects.png", dpi=180)
    plt.close(fig)
    return report


def correct_claim_family(reports: list[Path], output: Path):
    records = [json.loads(path.read_text()) for path in reports]
    q = bh_adjust([r["p_value"] for r in records])
    claims = []
    for path, record, adjusted in zip(reports, records, q, strict=True):
        eligible = record["effect"] >= record["minimum_effect"] and record["ci"][0] > 0
        claims.append(
            {
                "source": str(path),
                "source_sha256": file_hash(path),
                "q_value": adjusted,
                "statistical_gate": bool(eligible and adjusted < 0.05),
                "scientific_confirmation": "requires_validity_and_specificity_evidence",
            }
        )
    result = {"family_size": len(reports), "claims": claims}
    write_json(output, result, exclusive=True)
    return result


def mediation_summary(original: dict, intervened: dict, controls: list[dict], utility_pass: bool):
    suppression = original["effect"] - intervened["effect"]
    null = [original["effect"] - c["effect"] for c in controls]
    return {
        "raw_suppression": suppression,
        "intervention_mediated_fraction": stable_fraction(
            original["effect"], intervened["effect"], original["ci"]
        ),
        "exceeds_random_95th_percentile": bool(null and suppression > np.quantile(null, 0.95)),
        "utility_pass": utility_pass,
        "causal_specificity_confirmed": False,
        "note": "Run-paired intervention contrasts and registered multiplicity still required",
    }
