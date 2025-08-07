# Import pytest so we can use pytest.approx for floating point comparisons
import pytest

from solhunter_zero import investor_demo


def test_default_price_data_size() -> None:
    """ensure bundled investor demo data stays reasonably small"""
    prices, _ = investor_demo.load_prices()
    assert len(prices) <= 2000


def test_compute_weighted_returns_synthetic():
    prices = [10.0, 11.0, 9.0, 12.0]
    weights = {"buy_hold": 0.5, "momentum": 0.5}
    returns = investor_demo.compute_weighted_returns(prices, weights)
    assert len(returns) == 3
    assert returns[0] == pytest.approx(0.1, rel=1e-6)
    assert returns[1] == pytest.approx(-1 / 11, rel=1e-6)
    assert returns[2] == pytest.approx(1 / 3, rel=1e-6)


def test_compute_weighted_returns_negative_weights():
    prices = [10.0, 11.0, 9.0, 12.0]
    weights = {"buy_hold": 1.0, "mean_reversion": -0.5}
    returns = investor_demo.compute_weighted_returns(prices, weights)
    assert len(returns) == 3
    assert returns[0] == pytest.approx(1 / 15, rel=1e-6)
    assert returns[1] == pytest.approx(-2 / 11, rel=1e-6)
    assert returns[2] == pytest.approx(2 / 9, rel=1e-6)


def test_compute_weighted_returns_offsetting():
    prices = [10.0, 11.0, 9.0, 12.0]
    weights = {"buy_hold": 1.0, "momentum": -1.0}
    returns = investor_demo.compute_weighted_returns(prices, weights)
    assert returns[0] == pytest.approx(0.0, rel=1e-6)
    assert returns[1] == pytest.approx(-1 / 11, rel=1e-6)
    assert returns[2] == pytest.approx(0.0, rel=1e-6)


def test_max_drawdown_synthetic():
    returns = [0.1, -0.2, 0.05]
    assert investor_demo.max_drawdown(returns) == pytest.approx(0.2, rel=1e-6)
    assert investor_demo.max_drawdown([]) == 0.0


def test_correlations_synthetic(monkeypatch, tmp_path, dummy_mem):
    """Correlation pairs from synthetic price data."""

    prices = [1.0, 2.0, 1.0, 2.0]
    monkeypatch.setattr(
        investor_demo,
        "load_prices",
        lambda *_: (prices, ["2024-01-01"] * len(prices)),
    )

    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)

    async def fake_arbitrage() -> None:
        investor_demo.used_trade_types.add("arbitrage")

    async def fake_flash() -> None:
        investor_demo.used_trade_types.add("flash_loan")

    async def fake_sniper() -> None:
        investor_demo.used_trade_types.add("sniper")

    async def fake_dex() -> None:
        investor_demo.used_trade_types.add("dex_scanner")

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arbitrage)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash)
    monkeypatch.setattr(investor_demo, "_demo_sniper", fake_sniper)
    monkeypatch.setattr(investor_demo, "_demo_dex_scanner", fake_dex)

    captured: dict[tuple[str, str], float] = {}

    def fake_hedge(weights, corrs):
        captured.update(corrs)
        return weights

    monkeypatch.setattr(investor_demo, "hedge_allocation", fake_hedge)

    investor_demo.main(["--reports", str(tmp_path)])

    assert captured[("buy_hold", "momentum")] == pytest.approx(1.0, rel=1e-6)
    assert captured[("buy_hold", "mean_reversion")] == pytest.approx(-1.0, rel=1e-6)
    assert captured[("momentum", "mean_reversion")] == pytest.approx(-1.0, rel=1e-6)


def test_demo_trade_recorded(monkeypatch, tmp_path, dummy_mem):
    prices = [1.0, 2.0]
    dates = ["2024-01-01", "2024-01-02"]
    monkeypatch.setattr(investor_demo, "load_prices", lambda *_: (prices, dates))

    dummy = dummy_mem()
    monkeypatch.setattr(investor_demo, "Memory", lambda *a, **k: dummy)

    async def fake_arbitrage() -> None:
        investor_demo.used_trade_types.add("arbitrage")

    async def fake_flash() -> None:
        investor_demo.used_trade_types.add("flash_loan")

    async def fake_sniper() -> None:
        investor_demo.used_trade_types.add("sniper")

    async def fake_dex() -> None:
        investor_demo.used_trade_types.add("dex_scanner")

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arbitrage)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash)
    monkeypatch.setattr(investor_demo, "_demo_sniper", fake_sniper)
    monkeypatch.setattr(investor_demo, "_demo_dex_scanner", fake_dex)

    investor_demo.main(["--reports", str(tmp_path)])

    assert dummy.trade is not None
    trade = dummy.trade
    assert trade["token"] == "demo"
    assert trade["direction"] == "buy"
    assert trade["amount"] == 1.0
    assert trade["price"] == prices[0]
