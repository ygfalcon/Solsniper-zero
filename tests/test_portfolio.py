import pytest
from solhunter_zero.portfolio import Portfolio, calculate_order_size


@pytest.fixture(autouse=True)
def _no_fee(monkeypatch):
    monkeypatch.setattr("solhunter_zero.gas.get_current_fee", lambda testnet=False: 0.0)


def test_portfolio_update_and_pnl(tmp_path):
    path = tmp_path / "p.json"
    p = Portfolio(path=str(path))
    p.add("tok", 1, 1.0)
    p.update("tok", 1, 3.0)
    assert p.balances["tok"].amount == 2
    assert p.balances["tok"].entry_price == pytest.approx(2.0)
    pnl = p.unrealized_pnl({"tok": 4.0})
    assert pnl == pytest.approx(4.0)
    p.update("tok", -2, 4.0)
    assert "tok" not in p.balances


def test_portfolio_persistence(tmp_path):
    path = tmp_path / "pf.json"
    p1 = Portfolio(path=str(path))
    p1.add("tok", 1, 1.5)

    p2 = Portfolio(path=str(path))
    assert "tok" in p2.balances
    assert p2.balances["tok"].amount == 1
    assert p2.balances["tok"].entry_price == pytest.approx(1.5)


def test_position_roi():
    p = Portfolio(path=None)
    p.add("tok", 2, 1.0)
    roi = p.position_roi("tok", 1.5)
    assert roi == pytest.approx(0.5)


def test_calculate_order_size_basic():
    size = calculate_order_size(100.0, 1.0, risk_tolerance=0.1, max_allocation=0.2)
    assert size == pytest.approx(10.0)


def test_calculate_order_size_caps():
    size = calculate_order_size(100.0, 5.0, risk_tolerance=0.1, max_allocation=0.2)
    assert size == pytest.approx(10.0)


def test_calculate_order_size_negative_roi():
    assert calculate_order_size(100.0, -0.5) == 0.0


def test_calculate_order_size_risk_controls():
    base = calculate_order_size(
        100.0,
        1.0,
        volatility=0.0,
        drawdown=0.0,
        risk_tolerance=0.1,
        max_allocation=0.5,
        max_risk_per_token=0.5,
    )
    high_vol = calculate_order_size(
        100.0,
        1.0,
        volatility=1.0,
        drawdown=0.0,
        risk_tolerance=0.1,
        max_allocation=0.5,
        max_risk_per_token=0.5,
    )
    high_dd = calculate_order_size(
        100.0,
        1.0,
        volatility=0.0,
        drawdown=0.5,
        risk_tolerance=0.1,
        max_allocation=0.5,
        max_risk_per_token=0.5,
    )
    capped = calculate_order_size(
        100.0,
        10.0,
        volatility=0.0,
        drawdown=0.0,
        risk_tolerance=0.1,
        max_allocation=0.5,
        max_risk_per_token=0.1,
    )

    assert high_vol < base
    assert high_dd < base
    assert capped == pytest.approx(10.0)


def test_portfolio_drawdown():
    p = Portfolio(path=None)
    p.add("tok", 1, 1.0)
    p.update_drawdown({"tok": 1.0})
    assert p.current_drawdown({"tok": 1.0}) == pytest.approx(0.0)
    p.update_drawdown({"tok": 2.0})
    assert p.current_drawdown({"tok": 2.0}) == pytest.approx(0.0)
    p.update_drawdown({"tok": 1.0})
    assert p.current_drawdown({"tok": 1.0}) == pytest.approx(0.5)


def test_calculate_order_size_with_gas():
    size = calculate_order_size(
        100.0,
        1.0,
        risk_tolerance=0.1,
        max_allocation=0.2,
        gas_cost=1.0,
    )
    assert size == pytest.approx(9.0)


def test_percent_allocated():
    p = Portfolio(path=None)
    p.add("tok", 2, 1.0)
    p.add("oth", 3, 2.0)
    alloc = p.percent_allocated("tok")
    assert alloc == pytest.approx(2 / (2 + 6))
    alloc_prices = p.percent_allocated("tok", {"tok": 2.0, "oth": 2.0})
    assert alloc_prices == pytest.approx(4 / (4 + 6))


def test_order_size_respects_allocation():
    size = calculate_order_size(
        100.0,
        1.0,
        risk_tolerance=0.5,
        max_allocation=0.2,
        max_risk_per_token=0.5,
        current_allocation=0.15,
    )
    # Only 5% allocation left
    assert size == pytest.approx(5.0)

