import pytest

from cgl.forecasting import group_conformal_radius


def test_three_groups_do_not_manufacture_a_finite_ninety_percent_bound():
    assert group_conformal_radius([1.0, 2.0, 3.0], alpha=0.1) is None


def test_group_conformal_uses_finite_sample_order_statistic():
    assert group_conformal_radius(list(range(1, 10)), alpha=0.1) == 9
    assert group_conformal_radius(list(range(1, 20)), alpha=0.1) == 18


def test_conformal_rejects_negative_or_nonfinite_errors():
    with pytest.raises(ValueError):
        group_conformal_radius([float("nan")])
