import numpy as np
from solhunter_zero import models
import pytest


def test_train_transformer_model():
    prices = [1.0]
    for _ in range(20):
        prices.append(prices[-1] * 1.05)
    liquidity = np.linspace(10, 20, len(prices)).tolist()
    depth = np.linspace(1, 2, len(prices)).tolist()
    tx = np.linspace(5, 15, len(prices)).tolist()
    slippage = np.linspace(0.05, 0.1, len(prices)).tolist()
    volume = np.linspace(20, 40, len(prices)).tolist()

    model = models.train_transformer_model(
        prices,
        liquidity,
        depth,
        tx,
        slippage=slippage,
        volume=volume,
        seq_len=5,
        epochs=50,
    )
    seq = np.column_stack([
        prices[-5:],
        liquidity[-5:],
        depth[-5:],
        slippage[-5:],
        volume[-5:],
        tx[-5:],
    ])
    pred = model.predict(seq)
    assert pred == pytest.approx(0.05, abs=0.03)
