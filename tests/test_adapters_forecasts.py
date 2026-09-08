import pytest
import torch

from cgl.adapters import compose_factors
from cgl.forecasting import validate_forecast_data


def test_lora_composition_has_no_cross_terms():
    torch.manual_seed(2)
    a = [torch.randn(2, 7), torch.randn(3, 7)]
    b = [torch.randn(5, 2), torch.randn(5, 3)]
    weights, scales = [0.5, -0.2], [8.0, 3.0]
    aa, bb = compose_factors(a, b, weights, scales)
    expected = sum(w * s * (bi @ ai) for ai, bi, w, s in zip(a, b, weights, scales, strict=True))
    assert torch.allclose(bb @ aa, expected, atol=1e-5)


def test_forecast_rejects_preexposed_test_outcomes():
    train = [{"run_id": "a", "budget_fraction": 0.1, "features": {"loss": 1}, "target": 2}]
    pending = [{"run_id": "b", "budget_fraction": 0.1, "features": {"loss": 1}, "target": 3}]
    with pytest.raises(ValueError, match="outcomes"):
        validate_forecast_data(train, pending, ["loss"])


def test_forecast_rejects_final_checkpoint_features():
    pending = [{"run_id": "b", "budget_fraction": 1.0, "features": {"loss": 1}}]
    with pytest.raises(ValueError, match="Early"):
        validate_forecast_data([], pending, ["loss"])
