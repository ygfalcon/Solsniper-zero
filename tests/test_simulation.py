import numpy as np
import pytest

from solhunter_zero import simulation
from solhunter_zero.simulation import SimulationResult


def test_run_simulations_uses_metrics(monkeypatch):
    def fake_metrics(token):
        return {"mean": 0.01, "volatility": 0.0}

    captured = {}

    def fake_normal(mean, vol, days):
        captured["mean"] = mean
        captured["vol"] = vol
        return np.full(days, mean)

    monkeypatch.setattr(simulation, "fetch_token_metrics", fake_metrics)
    monkeypatch.setattr(simulation.np.random, "normal", fake_normal)

    results = simulation.run_simulations("tok", count=1, days=2)
    assert isinstance(results[0], SimulationResult)
    assert captured["mean"] == 0.01
    assert captured["vol"] == 0.0
    expected_roi = pytest.approx((1 + 0.01) ** 2 - 1)
    assert results[0].expected_roi == expected_roi
