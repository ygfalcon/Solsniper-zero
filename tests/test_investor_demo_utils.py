import pytest

from solhunter_zero import investor_demo


def test_compute_weighted_returns_synthetic():
    prices = [10.0, 11.0, 9.0, 12.0]
    weights = {"buy_hold": 0.5, "momentum": 0.5}
    returns = investor_demo.compute_weighted_returns(prices, weights)
    assert len(returns) == 3
    assert returns[0] == pytest.approx(0.1, rel=1e-6)
    assert returns[1] == pytest.approx(-1 / 11, rel=1e-6)
    assert returns[2] == pytest.approx(1 / 3, rel=1e-6)


def test_max_drawdown_synthetic():
    returns = [0.1, -0.2, 0.05]
    assert investor_demo.max_drawdown(returns) == pytest.approx(0.2, rel=1e-6)
    assert investor_demo.max_drawdown([]) == 0.0


def test_correlations_synthetic(monkeypatch, tmp_path):
    """Correlation pairs from synthetic price data."""

    prices = [1.0, 2.0, 1.0, 2.0]
    monkeypatch.setattr(investor_demo, "load_prices", lambda _=None: prices)

    class DummyMem:
        def __init__(self, *a, **k):
            pass

        def log_var(self, value: float):
            pass

        async def close(self) -> None:  # pragma: no cover - simple stub
            pass

    monkeypatch.setattr(investor_demo, "Memory", DummyMem)

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
    assert captured[("buy_hold", "mean_reversion")] == pytest.approx(
        -1.0, rel=1e-6
    )
    assert captured[("momentum", "mean_reversion")] == pytest.approx(
        -1.0, rel=1e-6
    )
