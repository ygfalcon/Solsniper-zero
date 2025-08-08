import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from solhunter_zero.portfolio import Portfolio
from solhunter_zero.strategy_manager import StrategyManager


def test_strategy_manager_showcase(monkeypatch):
    # Minimal configuration so importing strategies succeeds
    cfg_path = Path(__file__).resolve().parent / "demo_config.toml"
    cfg_path.write_text(
        """
solana_rpc_url = "https://api.mainnet-beta.solana.com"
dex_base_url = "https://quote-api.jup.ag"
agents = ["demo"]
[agent_weights]
demo = 1.0
"""
    )
    monkeypatch.setenv("SOLHUNTER_CONFIG", str(cfg_path))

    # Import strategies after config is set
    import solhunter_zero.sniper as sniper
    import solhunter_zero.arbitrage as arbitrage

    # Load deterministic price data
    data_path = Path(__file__).resolve().parents[1] / "data" / "prices_demo.json"
    price_data = json.loads(data_path.read_text())

    async def fake_fetch_prices(tokens):
        return {t: price_data.get(t, 0.0) for t in tokens}

    # Patch sniper dependencies for deterministic behaviour
    monkeypatch.setattr(sniper, "fetch_token_prices_async", fake_fetch_prices)
    monkeypatch.setattr(sniper, "run_simulations", lambda token, count=100: [
        SimpleNamespace(
            expected_roi=0.5,
            volatility=0.1,
            volume_spike=1.0,
            depth_change=0.0,
            whale_activity=0.0,
            tx_rate=1.0,
        )
    ])
    monkeypatch.setattr(sniper, "should_buy", lambda sims: True)
    monkeypatch.setattr(sniper, "should_sell", lambda *a, **k: False)
    monkeypatch.setattr(sniper, "dynamic_order_size", lambda *a, **k: 1.0)
    monkeypatch.setattr(sniper, "predict_price_movement", lambda token: 0.0)

    class DummyRiskManager:
        def __init__(self, **kwargs):
            self.risk_tolerance = kwargs.get("risk_tolerance", 0.1)
            self.max_allocation = kwargs.get("max_allocation", 1.0)
            self.max_risk_per_token = kwargs.get("max_risk_per_token", 1.0)
            self.max_drawdown = kwargs.get("max_drawdown", 1.0)
            self.volatility_factor = kwargs.get("volatility_factor", 1.0)
            self.risk_multiplier = kwargs.get("risk_multiplier", 1.0)
            self.min_portfolio_value = kwargs.get("min_portfolio_value", 0.0)

        def adjusted(self, **kwargs):
            return self

    monkeypatch.setattr(sniper, "RiskManager", DummyRiskManager)

    # Patch arbitrage dependencies
    async def fake_arbitrage(token, threshold, amount, dry_run=True):
        return {"token": token}

    monkeypatch.setattr(arbitrage, "detect_and_execute_arbitrage", fake_arbitrage)
    monkeypatch.setenv("ARBITRAGE_THRESHOLD", "0.01")
    monkeypatch.setenv("ARBITRAGE_AMOUNT", "2")

    portfolio = Portfolio(path=None)
    manager = StrategyManager()

    actions = asyncio.run(manager.evaluate("DEMO", portfolio))

    # Expect contributions from both strategies: one buy (sniper+arbitrage) and one sell (arbitrage)
    assert len(actions) == 2
    summary = {a["side"]: a for a in actions}
    assert summary["buy"]["amount"] == pytest.approx(3.0)
    assert summary["sell"]["amount"] == pytest.approx(2.0)
    assert summary["buy"]["price"] == pytest.approx(14.0)
