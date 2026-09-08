import pytest

from cgl.surface import audit_features, style_features


def test_surface_features_are_content_independent_counts():
    values = style_features("One sentence.\n- A second item!", token_count=9)
    assert values["tokens"] == 9
    assert values["newlines"] == 1
    assert values["bullets"] == 1


def test_identical_surface_pairs_pass_without_classifier_leakage():
    rows = []
    for i in range(30):
        for label in (0, 1):
            rows.append(
                {
                    "pair_id": str(i),
                    "label": label,
                    "features": style_features("A brief answer.", 4),
                }
            )
    report = audit_features(rows)
    assert report["surface_auc"] == 0.5
    assert report["surface_pass"]
    assert not set(report["train_pairs"]) & set(report["test_pairs"])


def test_length_confounded_corpus_fails():
    rows = []
    for i in range(30):
        for label in (0, 1):
            rows.append(
                {
                    "pair_id": str(i),
                    "label": label,
                    "features": style_features("Word. " * (1 + 9 * label), 2 + label * 20),
                }
            )
    assert not audit_features(rows)["surface_pass"]


def test_unpaired_features_are_invalid():
    with pytest.raises(ValueError, match="two"):
        audit_features([{"pair_id": "1", "label": 0, "features": style_features("a", 1)}])
