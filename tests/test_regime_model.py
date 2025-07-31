import pytest
from types import SimpleNamespace

from solhunter_zero.models.regime_model import train_regime_model, save_regime_model
from solhunter_zero.regime import load_regime_model as load_model_env, detect_regime_ml


@pytest.mark.slow
def test_regime_model_training(tmp_path):
    # create synthetic snapshots
    snaps = []
    price = 1.0
    for _ in range(40):
        snaps.append(SimpleNamespace(price=price, depth=1.0, sentiment=0.0))
        price *= 1.05
    for _ in range(40):
        snaps.append(SimpleNamespace(price=price, depth=1.0, sentiment=0.0))
        price *= 0.95
    for _ in range(40):
        snaps.append(SimpleNamespace(price=price, depth=1.0, sentiment=0.0))

    model = train_regime_model(snaps, seq_len=5, epochs=30)
    path = tmp_path / "regime.pt"
    save_regime_model(model, path)

    # load through regime module
    load_model_env(str(path))
    prices = [s.price for s in snaps[-6:-1]] + [snaps[-1].price * 1.05]
    features = [[1.0, 0.0] for _ in prices]
    regime = detect_regime_ml(prices, features)
    assert regime in {"bull", "bear", "sideways"}
