import numpy as np
import pytest

from solhunter_zero import simulation
from solhunter_zero.simulation import SimulationResult


def test_run_simulations_uses_metrics(monkeypatch):
    def fake_metrics(token):
        return {
            "mean": 0.01,
            "volatility": 0.0,
            "volume": 123.0,
            "liquidity": 456.0,
            "slippage": 0.01,
        }

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
    assert results[0].volume == pytest.approx(123.0)
    assert results[0].liquidity == pytest.approx(456.0)
    assert results[0].slippage == pytest.approx(0.01)


def test_fetch_token_metrics_base_url(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "mean_return": 0.1,
                "volatility": 0.03,
                "volume_24h": 321.0,
                "liquidity": 654.0,
                "slippage": 0.02,
            }

    captured = {}

    def fake_get(url, timeout=5):
        captured["url"] = url
        return FakeResp()

    monkeypatch.setenv("METRICS_BASE_URL", "http://metrics.local")
    monkeypatch.setattr(simulation.requests, "get", fake_get)

    metrics = simulation.fetch_token_metrics("tok")
    assert captured["url"] == "http://metrics.local/token/tok/metrics"
    assert metrics["mean"] == pytest.approx(0.1)
    assert metrics["volatility"] == pytest.approx(0.03)
    assert metrics["volume"] == pytest.approx(321.0)
    assert metrics["liquidity"] == pytest.approx(654.0)
    assert metrics["slippage"] == pytest.approx(0.02)


def test_run_simulations_volume_filter(monkeypatch):
    def fake_metrics(token):
        return {
            "mean": 0.01,
            "volatility": 0.02,
            "volume": 50.0,
            "liquidity": 100.0,
            "slippage": 0.05,
        }

    monkeypatch.setattr(simulation, "fetch_token_metrics", fake_metrics)

    results = simulation.run_simulations("tok", count=1, min_volume=100.0)
    assert results == []
