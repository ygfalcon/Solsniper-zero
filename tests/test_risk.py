import pytest

from solhunter_zero.risk import RiskManager
from solhunter_zero.portfolio import calculate_order_size


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
    low = rm.adjusted(0.0, 0.0, depth_change=-1.0, whale_activity=1.0)
    assert high.risk_tolerance > base.risk_tolerance
    assert low.risk_tolerance < base.risk_tolerance
