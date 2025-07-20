import sys
import types
import asyncio
from solhunter_zero.strategy_manager import StrategyManager


class DummyPortfolio:
    pass


def test_strategy_manager_invokes_modules(monkeypatch):
    calls = []

    async def eval1(token, portfolio):
        calls.append("s1")
        return [{"token": token, "side": "buy", "amount": 1, "price": 0}]

    def eval2(token, portfolio):
        calls.append("s2")
        return [{"token": token, "side": "sell", "amount": 1, "price": 0}]

    mod1 = types.SimpleNamespace(evaluate=eval1)
    mod2 = types.SimpleNamespace(evaluate=eval2)
    monkeypatch.setitem(sys.modules, "mod1", mod1)
    monkeypatch.setitem(sys.modules, "mod2", mod2)

    mgr = StrategyManager(["mod1", "mod2"])
    actions = asyncio.run(mgr.evaluate("tok", DummyPortfolio()))

    assert "s1" in calls and "s2" in calls
    assert {"token": "tok", "side": "buy", "amount": 1, "price": 0} in actions
    assert {"token": "tok", "side": "sell", "amount": 1, "price": 0} in actions


def test_strategy_manager_merges_actions(monkeypatch):
    async def eval1(token, portfolio):
        return [{"token": token, "side": "buy", "amount": 1, "price": 1.0}]

    async def eval2(token, portfolio):
        return [{"token": token, "side": "buy", "amount": 1, "price": 3.0}]

    mod1 = types.SimpleNamespace(evaluate=eval1)
    mod2 = types.SimpleNamespace(evaluate=eval2)
    monkeypatch.setitem(sys.modules, "mod1", mod1)
    monkeypatch.setitem(sys.modules, "mod2", mod2)

    mgr = StrategyManager(["mod1", "mod2"])
    actions = asyncio.run(mgr.evaluate("tok", DummyPortfolio()))

    assert actions == [{"token": "tok", "side": "buy", "amount": 2.0, "price": 2.0}]
