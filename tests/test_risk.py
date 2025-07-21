import pytest

from solhunter_zero.risk import RiskManager
from solhunter_zero.portfolio import calculate_order_size


@pytest.fixture(autouse=True)
def _no_fee(monkeypatch):
    monkeypatch.setattr("solhunter_zero.gas.get_current_fee", lambda testnet=False: 0.0)


def test_risk_manager_adjustments_reduce_size():
    rm = RiskManager(
        risk_tolerance=0.1,
        max_allocation=0.5,
        max_risk_per_token=0.5,
        max_drawdown=1.0,
        volatility_factor=1.0,
    )
    base = rm.adjusted(0.0, 0.0)
    base_size = calculate_order_size(
        100.0,
        1.0,
        0.0,
        0.0,
        risk_tolerance=base.risk_tolerance,
        max_allocation=base.max_allocation,
        max_risk_per_token=base.max_risk_per_token,
    )
    adjusted = rm.adjusted(drawdown=0.5, volatility=1.0)
    adj_size = calculate_order_size(
        100.0,
        1.0,
        0.0,
        0.0,
        risk_tolerance=adjusted.risk_tolerance,
        max_allocation=adjusted.max_allocation,
        max_risk_per_token=adjusted.max_risk_per_token,
    )
    assert adj_size < base_size


def test_risk_multiplier_increases_size():
    rm = RiskManager(
        risk_tolerance=0.1,
        max_allocation=0.2,
        max_risk_per_token=0.2,
        risk_multiplier=2.0,
    )
    params = rm.adjusted(0.0, 0.0)
    size = calculate_order_size(
        100.0,
        1.0,
        0.0,
        0.0,
        risk_tolerance=params.risk_tolerance,
        max_allocation=params.max_allocation,
        max_risk_per_token=params.max_risk_per_token,
    )
    assert size > 0.0
    assert params.risk_tolerance > 0.1


def test_risk_manager_new_metrics():
    rm = RiskManager(risk_tolerance=0.1, max_allocation=0.2, max_risk_per_token=0.2)
    base = rm.adjusted(0.0, 0.0)
    high = rm.adjusted(0.0, 0.0, volume_spike=2.0)
    low = rm.adjusted(0.0, 0.0, depth_change=-1.0, whale_activity=1.0, tx_rate=0.5)
    burst = rm.adjusted(0.0, 0.0, tx_rate=2.0)
    assert high.risk_tolerance > base.risk_tolerance
    assert low.risk_tolerance < base.risk_tolerance
    assert burst.risk_tolerance > base.risk_tolerance


def test_low_portfolio_scales_risk():
    rm = RiskManager(
        risk_tolerance=0.1,
        max_allocation=0.2,
        max_risk_per_token=0.2,
        min_portfolio_value=20.0,
    )
    high = rm.adjusted(0.0, 0.0, portfolio_value=100.0)
    low = rm.adjusted(0.0, 0.0, portfolio_value=10.0)
    assert low.risk_tolerance < high.risk_tolerance
    size_low = calculate_order_size(
        10.0,
        1.0,
        0.0,
        0.0,
        risk_tolerance=low.risk_tolerance,
        max_allocation=low.max_allocation,
        max_risk_per_token=low.max_risk_per_token,
        min_portfolio_value=low.min_portfolio_value,
    )
    assert size_low < calculate_order_size(
        100.0,
        1.0,
        0.0,
        0.0,
        risk_tolerance=high.risk_tolerance,
        max_allocation=high.max_allocation,
        max_risk_per_token=high.max_risk_per_token,
        min_portfolio_value=high.min_portfolio_value,
    )


def test_extreme_metric_values():
    rm = RiskManager(risk_tolerance=0.1, max_allocation=0.2, max_risk_per_token=0.2)
    base = rm.adjusted(0.0, 0.0)

    high_rate = rm.adjusted(0.0, 0.0, tx_rate=5.0)
    assert high_rate.risk_tolerance > base.risk_tolerance

    deep = rm.adjusted(0.0, 0.0, depth_change=3.0)
    assert deep.risk_tolerance < base.risk_tolerance

    whales = rm.adjusted(0.0, 0.0, whale_activity=5.0)
    assert whales.risk_tolerance < base.risk_tolerance
