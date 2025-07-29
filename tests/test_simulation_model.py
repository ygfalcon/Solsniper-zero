import numpy as np
import torch
import pytest

from solhunter_zero import models, simulation


def setup_function(_):
    simulation.invalidate_simulation_models()


def test_run_simulations_uses_model(tmp_path, monkeypatch):
    model = models.PriceModel(input_dim=4, hidden_dim=4, num_layers=1)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.fc.bias.fill_(0.1)
    model_path = tmp_path / "model.pt"
    models.save_model(model, model_path)

    metrics = {
        "mean": 0.0,
        "volatility": 0.0,
        "volume": 10.0,
        "liquidity": 20.0,
        "slippage": 0.01,
        "depth": 1.0,
        "price_history": [1.0, 1.1, 1.2],
        "liquidity_history": [10.0, 11.0, 12.0],
        "depth_history": [1.0, 1.1, 1.2],
        "slippage_history": [0.01, 0.01, 0.01],
        "tx_count_history": [5, 6, 7],
    }

    monkeypatch.setattr(simulation, "fetch_token_metrics", lambda _t: metrics)
    monkeypatch.setattr(
        simulation.onchain_metrics, "fetch_dex_metrics_async", lambda _t: {}
    )
    monkeypatch.setattr(simulation.np.random, "normal", lambda mean, vol, size: np.full(size, mean))
    monkeypatch.setenv("PRICE_MODEL_PATH", str(model_path))

    res = simulation.run_simulations("tok", count=1, days=2)[0]
    assert res.expected_roi > 0.2
