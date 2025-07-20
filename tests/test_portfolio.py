import pytest
from solhunter_zero.portfolio import Portfolio


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

