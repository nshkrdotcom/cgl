import pytest

from cgl.evaluation import numeric_judgment
from cgl.measurement import missingness_bounds


def test_unparsed_judge_outputs_bound_rates_instead_of_disappearing():
    assert missingness_bounds(
        [{"misaligned": True}, {"misaligned": False}, {"misaligned": None}, {}]
    ) == {"lower": 0.25, "upper": 0.75, "missing": 2, "total": 4}


def test_no_evidence_is_not_a_zero_misalignment_rate():
    with pytest.raises(ValueError):
        missingness_bounds([])


def test_rating_parser_preserves_refusals_and_rejects_explanatory_numbers():
    assert numeric_judgment("30.5") == 30.5
    assert numeric_judgment("REFUSAL") == "REFUSAL"
    for text in ("This is 30", "101", "-1", "I rate it 50 out of 100"):
        with pytest.raises(ValueError):
            numeric_judgment(text)
