"""Inference at the independent training-run level, with paired prompt sampling."""

import itertools

import numpy as np


def sign_flip_p(effects, *, two_sided=False, seed=0, draws=100_000) -> float:
    x = np.asarray(effects, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite, nonempty run-level effects")
    observed = abs(x.mean()) if two_sided else x.mean()
    exact = len(x) <= 18
    signs = (
        np.asarray(list(itertools.product((-1, 1), repeat=len(x))))
        if exact
        else np.random.default_rng(seed).choice((-1, 1), size=(draws, len(x)))
    )
    null = (signs * x).mean(axis=1)
    if two_sided:
        null = np.abs(null)
    count = int(np.count_nonzero(null >= observed - 1e-12))
    return float(count / len(null) if exact else (count + 1) / (len(null) + 1))


def bh_adjust(p_values) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    if not len(p):
        return []
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(p)
    values = p[order] * len(p) / np.arange(1, len(p) + 1)
    values = np.minimum.accumulate(values[::-1])[::-1].clip(0, 1)
    output = np.empty_like(values)
    output[order] = values
    return output.tolist()


def paired_effect(aligned, misaligned, *, bootstrap=10_000, seed=0, minimum=0.15) -> dict:
    a, b = np.asarray(aligned, dtype=float), np.asarray(misaligned, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or min(a.shape) == 0:
        raise ValueError("Expected paired [run, prompt] arrays of identical nonzero shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Missing or non-finite measurements must be resolved explicitly")
    difference = b - a
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap)
    for i in range(bootstrap):
        runs = rng.integers(len(a), size=len(a))
        prompts = rng.integers(a.shape[1], size=a.shape[1])
        means[i] = difference[np.ix_(runs, prompts)].mean()
    ci = np.quantile(means, [0.025, 0.975]).tolist()
    effects = difference.mean(axis=1)
    p = sign_flip_p(effects)
    return {
        "effect": float(effects.mean()),
        "run_effects": effects.tolist(),
        "ci": ci,
        "p_value": p,
        "n_runs": len(a),
        "n_prompts": a.shape[1],
        "minimum_effect": minimum,
        # This is single-comparison eligibility; family-level q is applied separately.
        "single_comparison_gate": bool(p < 0.05 and ci[0] > 0 and effects.mean() >= minimum),
        "multiplicity_status": "requires_registered_family_correction",
        "bootstrap_seed": seed,
        "bootstrap_samples": bootstrap,
    }


def cluster_rate(records: list[dict], field="misaligned", *, seed=0, bootstrap=10_000):
    groups = {}
    for row in records:
        if row.get(field) is not None:
            groups.setdefault(row["prompt_id"], []).append(float(row[field]))
    if not groups:
        raise ValueError("No valid scored prompt clusters")
    means = np.asarray([np.mean(groups[key]) for key in sorted(groups)])
    rng = np.random.default_rng(seed)
    samples = rng.choice(means, (bootstrap, len(means))).mean(axis=1)
    return {
        "mean": float(means.mean()),
        "ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "n_prompt_clusters": len(means),
        "n_scored": sum(map(len, groups.values())),
        "prompt_rates": dict(zip(sorted(groups), means.tolist(), strict=True)),
        "scope": "conditional_on_this_training_run",
    }


def stable_fraction(total: float, after: float, total_ci) -> float | None:
    if total <= 0 or total_ci[0] <= 0:
        return None
    return float((total - after) / total)


def power_simulation(effect=0.15, run_sd=0.1, seeds=(3, 6, 8, 12, 16), draws=500):
    rng = np.random.default_rng(42)
    result = []
    for n in seeds:
        detected = sum(sign_flip_p(rng.normal(effect, run_sd, n)) < 0.05 for _ in range(draws))
        result.append(
            {"runs_per_condition": n, "power": detected / draws, "minimum_one_sided_p": 2.0**-n}
        )
    return {
        "effect": effect,
        "between_run_sd": run_sd,
        "simulations": draws,
        "multiplicity": "unadjusted; plan family before confirmation",
        "estimates": result,
    }
