import numpy as np
from solhunter_zero import models
import pytest


def test_train_price_model_converges():
    prices = [1.0]
    for _ in range(20):
        prices.append(prices[-1] * 1.1)
    liquidity = np.linspace(100, 120, len(prices)).tolist()
    depth = np.linspace(1, 2, len(prices)).tolist()
    tx = np.linspace(10, 30, len(prices)).tolist()

    model = models.train_price_model(prices, liquidity, depth, tx, seq_len=5, epochs=100)
    seq = np.column_stack([prices[-5:], liquidity[-5:], depth[-5:], tx[-5:]])
    pred = model.predict(seq)
    assert pred == pytest.approx(0.1, abs=0.02)
