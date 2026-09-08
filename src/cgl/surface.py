"""Paired presentation controls with grouped, quantitative surface audits."""

import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def style_features(text: str, token_count: int) -> dict[str, float]:
    words = re.findall(r"\b\w+\b", text.lower())
    count = max(len(text), 1)
    return {
        "tokens": float(token_count),
        "characters": float(len(text)),
        "sentences": float(len(re.findall(r"[.!?]+", text))),
        "newlines": float(text.count("\n")),
        "bullets": float(len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s", text))),
        "headings": float(len(re.findall(r"(?m)^#+\s", text))),
        "code_fences": float(text.count("```")),
        "commas": float(text.count(",")),
        "semicolons": float(text.count(";")),
        "colons": float(text.count(":")),
        "questions": float(text.count("?")),
        "exclamations": float(text.count("!")),
        "uppercase_fraction": sum(c.isupper() for c in text) / count,
        "digit_fraction": sum(c.isdigit() for c in text) / count,
        "mean_word_length": sum(map(len, words)) / max(len(words), 1),
        "type_token_ratio": len(set(words)) / max(len(words), 1),
        "hedges": float(
            sum(w in {"may", "might", "perhaps", "possibly", "usually"} for w in words)
        ),
        "directives": float(sum(w in {"must", "always", "never", "should"} for w in words)),
    }


def audit_features(rows: list[dict], seed=0) -> dict:
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    if any(len(pair) != 2 or {r["label"] for r in pair} != {0, 1} for pair in by_pair.values()):
        raise ValueError("Each pair requires exactly two opposite-label observations")
    if len(by_pair) < 10:
        raise ValueError("At least ten independent pairs required for a surface audit")
    names = sorted(rows[0]["features"])
    x = np.asarray([[row["features"][name] for name in names] for row in rows])
    y = np.asarray([row["label"] for row in rows])
    groups = np.asarray([row["pair_id"] for row in rows])
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(x, y, groups)
    )
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1, random_state=seed))
    classifier.fit(x[train], y[train])
    auc = float(roc_auc_score(y[test], classifier.predict_proba(x[test])[:, 1]))
    accuracy = float(balanced_accuracy_score(y[test], classifier.predict(x[test])))
    a, m = x[y == 0], x[y == 1]
    denominator = np.sqrt((a.var(axis=0, ddof=1) + m.var(axis=0, ddof=1)) / 2)
    smd = np.divide(m.mean(0) - a.mean(0), np.maximum(denominator, 1e-12))
    lengths, sentences, structures = [], [], []
    for pair in by_pair.values():
        af, mf = pair[0]["features"], pair[1]["features"]
        lengths.append(
            abs(af["tokens"] - mf["tokens"]) <= 0.05 * max((af["tokens"] + mf["tokens"]) / 2, 1)
        )
        sentences.append(abs(af["sentences"] - mf["sentences"]) <= 1)
        structures.append(all(af[key] == mf[key] for key in ["bullets", "headings", "code_fences"]))
    passed = (
        max(auc, 1 - auc) <= 0.60
        and max(accuracy, 1 - accuracy) <= 0.60
        and float(np.abs(smd).max()) <= 0.25
        and np.mean(lengths) >= 0.95
        and np.mean(sentences) >= 0.95
        and np.mean(structures) >= 0.98
    )
    return {
        "pairs": len(by_pair),
        "surface_auc": auc,
        "balanced_accuracy": accuracy,
        "smd": dict(zip(names, smd.tolist(), strict=True)),
        "length_match_fraction": float(np.mean(lengths)),
        "length_control_pass": bool(np.mean(lengths) >= 0.95),
        "sentence_match_fraction": float(np.mean(sentences)),
        "structure_match_fraction": float(np.mean(structures)),
        "surface_pass": bool(passed),
        "train_pairs": sorted(set(groups[train])),
        "test_pairs": sorted(set(groups[test])),
        "human_fidelity_verified": False,
    }
