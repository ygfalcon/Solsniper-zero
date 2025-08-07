import asyncio
import os
import sys
import types
import pytest

from solhunter_zero.portfolio import Portfolio
from solhunter_zero.strategy_manager import StrategyManager
from solhunter_zero.simulation import SimulationResult


def _install_stub_modules(monkeypatch):
    # Create sniper stub with deterministic helpers
    sniper = types.ModuleType("solhunter_zero.sniper")

    def run_simulations(token, count=100):
        return [SimulationResult(1.0, 1.0)]

    def should_buy(sims):
        return True

    def should_sell(sims, **kw):
        return False

    async def fetch_token_prices_async(tokens):
        return {t: 1.0 for t in tokens}

    def predict_price_movement(token):
        return 0.1

    def dynamic_order_size(*args, **kwargs):
        return 1.0

    async def evaluate(token, portfolio):
        prices = await fetch_token_prices_async({token})
        amount = dynamic_order_size()
        return [{"token": token, "side": "buy", "amount": amount, "price": prices[token]}]

    sniper.run_simulations = run_simulations
    sniper.should_buy = should_buy
    sniper.should_sell = should_sell
    sniper.fetch_token_prices_async = fetch_token_prices_async
    sniper.predict_price_movement = predict_price_movement
    sniper.dynamic_order_size = dynamic_order_size
    sniper.evaluate = evaluate

    monkeypatch.setitem(sys.modules, "solhunter_zero.sniper", sniper)

    # Create arbitrage stub that yields a profitable opportunity
    arbitrage = types.ModuleType("solhunter_zero.arbitrage")

    async def detect_and_execute_arbitrage(token, **kwargs):
        return (0, 1)

    async def evaluate(token, portfolio):
        threshold = float(os.getenv("ARBITRAGE_THRESHOLD", "0") or 0)
        amount = float(os.getenv("ARBITRAGE_AMOUNT", "0") or 0)
        if threshold <= 0 or amount <= 0:
            return []
        res = await detect_and_execute_arbitrage(token, threshold=threshold, amount=amount, dry_run=True)
        if not res:
            return []
        action = {"token": token, "amount": amount, "price": 0.0}
        return [dict(action, side="buy"), dict(action, side="sell")]

    arbitrage.detect_and_execute_arbitrage = detect_and_execute_arbitrage
    arbitrage.evaluate = evaluate

    monkeypatch.setitem(sys.modules, "solhunter_zero.arbitrage", arbitrage)


def test_one_click_all_strategies(monkeypatch):
    _install_stub_modules(monkeypatch)

    monkeypatch.setenv("ARBITRAGE_THRESHOLD", "0.01")
    monkeypatch.setenv("ARBITRAGE_AMOUNT", "2")

    portfolio = Portfolio(path=None)
    mgr = StrategyManager()

    actions = asyncio.run(mgr.evaluate("TKN", portfolio))

    assert len(actions) == 2
    merged = {a["side"]: a for a in actions}
    assert merged["buy"]["amount"] == pytest.approx(3.0)
    assert merged["buy"]["price"] == pytest.approx(1 / 3)
    assert merged["sell"]["amount"] == pytest.approx(2.0)
    assert merged["sell"]["price"] == 0.0
