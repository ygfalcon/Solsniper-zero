import numpy as np

from solhunter_zero.investor_demo import compute_weighted_returns
from solhunter_zero.backtester import buy_and_hold, momentum


def test_single_strategy_weighted_returns_expected():
    prices = [100, 110, 90, 95]
    weights = {"buy_hold": 1.0}
    result = compute_weighted_returns(prices, weights)
    expected = np.array(buy_and_hold(prices), dtype=float)
    assert len(result) == len(expected)
    assert np.allclose(result, expected)


def test_mixed_weights_distribute_returns():
    prices = [100, 110, 90, 95]
    weights = {"buy_hold": 0.75, "momentum": 0.25}
    result = compute_weighted_returns(prices, weights)
    bh = np.array(buy_and_hold(prices), dtype=float)
    mom = np.array(momentum(prices), dtype=float)
    min_len = min(len(bh), len(mom))
    expected = (0.75 * bh[:min_len] + 0.25 * mom[:min_len]) / (0.75 + 0.25)
    assert np.allclose(result, expected)


def test_zero_or_empty_weights_yield_empty_array():
    prices = [100, 110, 120]
    zero_weights = {"buy_hold": 0.0, "momentum": 0.0}
    assert compute_weighted_returns(prices, zero_weights).size == 0
    assert compute_weighted_returns(prices, {}).size == 0
