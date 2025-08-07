import asyncio
import runpy
import sys
import types
from pathlib import Path

from solhunter_zero.strategy_manager import StrategyManager


# ---------------------------------------------------------------------------

def test_investor_showcase(monkeypatch, capsys):
    """Execute investor_showcase script and ensure default strategies appear."""

    # Stub out modules that would otherwise trigger heavy imports
    psm = types.ModuleType("solhunter_zero.price_stream_manager")
    psm.PriceStreamManager = object
    monkeypatch.setitem(sys.modules, "solhunter_zero.price_stream_manager", psm)

    from solhunter_zero.investor_demo import DEFAULT_STRATEGIES as demo_strats
    names = [name for name, _ in demo_strats]

    # Align StrategyManager defaults with the demo strategies
    monkeypatch.setattr(StrategyManager, "DEFAULT_STRATEGIES", names, raising=False)

    agents_mod = types.ModuleType("solhunter_zero.agents")

    class BaseAgent:
        def __init__(self, name: str | None = None):
            self.name = name or self.__class__.__name__

        async def propose_trade(self, token, portfolio, *, depth=None, imbalance=None):
            return []

    agents_mod.BaseAgent = BaseAgent
    monkeypatch.setitem(sys.modules, "solhunter_zero.agents", agents_mod)

    mem_mod = types.ModuleType("solhunter_zero.agents.memory")

    class MemoryAgent(BaseAgent):
        def __init__(self, memory):
            self.memory = memory

    mem_mod.MemoryAgent = MemoryAgent
    monkeypatch.setitem(sys.modules, "solhunter_zero.agents.memory", mem_mod)

    agent_manager_mod = types.ModuleType("solhunter_zero.agent_manager")

    def _get(obj, field, default=None):
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)

    class AgentManager:
        def __init__(self, agents, memory_agent=None, weights=None):
            self.agents = list(agents)
            self.memory_agent = memory_agent
            self.weights = dict(weights or {})

    class StrategySelector:
        def __init__(self, memory_agent):
            self.memory_agent = memory_agent

        def weight_agents(self, agents, base_weights):
            trades_coro = self.memory_agent.memory.list_trades(limit=1000)
            trades = (
                asyncio.run(trades_coro)
                if asyncio.iscoroutine(trades_coro)
                else trades_coro
            )
            summary = {a.name: {"buy": 0.0, "sell": 0.0} for a in agents}
            for t in trades:
                reason = _get(t, "reason", "")
                side = _get(t, "direction", "")
                amt = float(_get(t, "amount", 0.0))
                price = float(_get(t, "price", 0.0))
                info = summary.setdefault(reason, {"buy": 0.0, "sell": 0.0})
                info[side] += amt * price
            rois = {}
            for name, info in summary.items():
                spent = info.get("buy", 0.0)
                revenue = info.get("sell", 0.0)
                if spent > 0:
                    rois[name] = (revenue - spent) / spent
            if not rois:
                return list(agents), dict(base_weights)
            max_roi = max(rois.values())
            min_roi = min(rois.values())
            if max_roi == min_roi:
                return list(agents), dict(base_weights)
            weights = {}
            for a in agents:
                roi = rois.get(a.name, 0.0)
                norm = (roi - min_roi) / (max_roi - min_roi) if max_roi != min_roi else 0.0
                weights[a.name] = base_weights.get(a.name, 1.0) * (1.0 + norm)
            return list(agents), weights

    agent_manager_mod.AgentManager = AgentManager
    agent_manager_mod.StrategySelector = StrategySelector
    monkeypatch.setitem(sys.modules, "solhunter_zero.agent_manager", agent_manager_mod)

    runpy.run_path(str(Path("scripts/investor_showcase.py")), run_name="__main__")

    out = capsys.readouterr().out
    for name in StrategyManager.DEFAULT_STRATEGIES:
        assert name in out
    assert "Iteration 3" in out
