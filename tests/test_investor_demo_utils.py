import pytest

from solhunter_zero import investor_demo


def test_compute_weighted_returns_synthetic():
    prices = [10.0, 11.0, 9.0, 12.0]
    weights = {"buy_hold": 0.5, "momentum": 0.5}
    returns = investor_demo.compute_weighted_returns(prices, weights)
    assert len(returns) == 2
    assert returns[0] == pytest.approx(0.1, rel=1e-6)
    assert returns[1] == pytest.approx(5 / 66, rel=1e-6)


def test_max_drawdown_synthetic():
    returns = [0.1, -0.2, 0.05]
    assert investor_demo.max_drawdown(returns) == pytest.approx(0.2, rel=1e-6)
    assert investor_demo.max_drawdown([]) == 0.0
