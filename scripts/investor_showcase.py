#!/usr/bin/env python
"""Demonstrate AgentManager weight updates with dummy strategy data."""

"""Lightweight showcase of dynamic strategy weighting."""

import asyncio
import types
import sys
from typing import List, Dict

# ``AgentManager`` pulls in heavy dependencies such as network clients and
# Solana libraries.  For this standalone demo we stub out the modules the
# manager imports so that only the weight-management logic is exercised.


class _StubExec:
    name = "execution"

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - demo stub
        pass

    async def propose_trade(self, *args, **kwargs):  # pragma: no cover - demo
        return []

    def add_executor(self, *args, **kwargs) -> None:  # pragma: no cover - demo
        return None

    def close(self) -> None:  # pragma: no cover - demo
        return None


sys.modules.setdefault(
    "solhunter_zero.agents.execution", types.SimpleNamespace(ExecutionAgent=_StubExec)
)
sys.modules.setdefault(
    "solhunter_zero.agents.discovery", types.SimpleNamespace(DiscoveryAgent=object)
)
sys.modules.setdefault(
    "solhunter_zero.agents.swarm", types.SimpleNamespace(AgentSwarm=object)
)
sys.modules.setdefault(
    "solhunter_zero.agents.emotion_agent", types.SimpleNamespace(EmotionAgent=object)
)
sys.modules.setdefault(
    "solhunter_zero.agents.rl_weight_agent", types.SimpleNamespace(RLWeightAgent=object)
)
sys.modules.setdefault(
    "solhunter_zero.agents.hierarchical_rl_agent", types.SimpleNamespace(HierarchicalRLAgent=object)
)
sys.modules.setdefault(
    "solhunter_zero.agents.attention_swarm",
    types.SimpleNamespace(AttentionSwarm=object, load_model=lambda *a, **k: None),
)
sys.modules.setdefault(
    "solhunter_zero.rl_training", types.SimpleNamespace(MultiAgentRL=object)
)
sys.modules.setdefault(
    "solhunter_zero.multi_rl", types.SimpleNamespace(PopulationRL=object)
)
sys.modules.setdefault(
    "solhunter_zero.swarm_coordinator",
    types.SimpleNamespace(
        SwarmCoordinator=lambda mem, weights, regime_weights=None: types.SimpleNamespace(
            base_weights=weights
        ),
    ),
)
sys.modules.setdefault(
    "solhunter_zero.device",
    types.SimpleNamespace(
        get_default_device=lambda: "cpu",
        detect_gpu=lambda: False,
        get_gpu_backend=lambda: "none",
    ),
)
sys.modules.setdefault(
    "solhunter_zero.hierarchical_rl", types.SimpleNamespace(SupervisorAgent=object)
)
sys.modules.setdefault(
    "solhunter_zero.regime", types.SimpleNamespace(detect_regime=lambda *a, **k: None)
)
sys.modules.setdefault(
    "solhunter_zero.datasets.sample_ticks",
    types.SimpleNamespace(load_sample_ticks=lambda *a, **k: [], DEFAULT_PATH=""),
)
sys.modules.setdefault(
    "solhunter_zero.backtest_cli", types.SimpleNamespace(bayesian_optimize_weights=lambda *a, **k: {})
)
sys.modules.setdefault(
    "solhunter_zero.price_stream_manager", types.SimpleNamespace(PriceStreamManager=object)
)

from solhunter_zero.agent_manager import AgentManager, StrategySelector
from solhunter_zero.agents import BaseAgent
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.simple_memory import SimpleMemory
from solhunter_zero.investor_demo import DEFAULT_STRATEGIES


class DummyStrategy(BaseAgent):
    """Minimal agent representing a trading strategy."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def propose_trade(
        self, token: str, portfolio, *, depth=None, imbalance=None
    ) -> List[Dict[str, float]]:
        return []


def compute_roi(prices: List[float], strat) -> float:
    """Return cumulative ROI for ``strat`` over ``prices``."""
    returns = strat(prices)
    total = 1.0
    for r in returns:
        total *= 1.0 + float(r)
    return total - 1.0


async def log_roi(mem: SimpleMemory, name: str, roi: float) -> None:
    """Record synthetic trades yielding ``roi`` for strategy ``name``."""
    await mem.log_trade(
        token="demo", direction="buy", amount=1.0, price=1.0, reason=name
    )
    await mem.log_trade(
        token="demo", direction="sell", amount=1.0, price=1.0 + roi, reason=name
    )


def main() -> None:
    """Run a simple backtest loop adjusting weights after each iteration."""
    mem = SimpleMemory()
    mem_agent = MemoryAgent(mem)

    # Wrap ``list_trades`` so that StrategySelector receives objects with
    # attribute access, mirroring the SQLAlchemy model used in production.
    orig_list_trades = mem.list_trades

    async def _list_trades(*args, **kwargs):
        trades = await orig_list_trades(*args, **kwargs)
        return [types.SimpleNamespace(**t) for t in trades]

    mem.list_trades = _list_trades  # type: ignore[assignment]

    agents = [DummyStrategy(name) for name, _ in DEFAULT_STRATEGIES]
    weights = {agent.name: 1.0 for agent in agents}
    manager = AgentManager(agents, memory_agent=mem_agent, weights=weights)
    selector = StrategySelector(mem_agent)

    price_windows = [
        [100, 105, 102, 108, 110],
        [110, 115, 117, 120, 125],
        [125, 123, 128, 130, 135],
    ]

    for i, prices in enumerate(price_windows, 1):
        rois: Dict[str, float] = {}
        for name, strat in DEFAULT_STRATEGIES:
            roi = compute_roi(prices, strat)
            rois[name] = roi
            asyncio.run(log_roi(mem, name, roi))

        _, new_weights = selector.weight_agents(manager.agents, manager.weights)
        manager.weights = new_weights

        print(f"Iteration {i}:")
        for name, roi in rois.items():
            weight = manager.weights.get(name, 0.0)
            print(f"  {name:15s} ROI={roi: .4f}  weight={weight: .3f}")
        print()


if __name__ == "__main__":
    main()
