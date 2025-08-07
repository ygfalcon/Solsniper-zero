from __future__ import annotations

import importlib.metadata
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Type

from ..portfolio import Portfolio

if TYPE_CHECKING:  # Imports for type checking only to avoid circular imports
    from .alien_cipher_agent import AlienCipherAgent
    from .arbitrage import ArbitrageAgent
    from .conviction import ConvictionAgent
    from .discovery import DiscoveryAgent
    from .dqn import DQNAgent
    from .emotion_agent import EmotionAgent
    from .execution import ExecutionAgent
    from .exit import ExitAgent
    from .flashloan_sandwich import FlashloanSandwichAgent
    from .llm_reasoner import LLMReasoner
    from .memory import MemoryAgent
    from .mempool_sniper import MempoolSniperAgent
    from .meta_conviction import MetaConvictionAgent
    from .mev_sandwich import MEVSandwichAgent
    from .momentum import MomentumAgent
    from .opportunity_cost import OpportunityCostAgent
    from .portfolio_agent import PortfolioAgent
    from .ppo_agent import PPOAgent
    from .ramanujan_agent import RamanujanAgent
    from .sac_agent import SACAgent
    from .simulation import SimulationAgent
    from .strange_attractor import StrangeAttractorAgent


class BaseAgent(ABC):
    """Abstract trading agent."""

    name: str = "base"

    @abstractmethod
    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        """Return proposed trade actions for ``token``."""
        raise NotImplementedError

    def explain_proposal(
        self,
        actions: List[Dict[str, Any]] | None = None,
        token: str | None = None,
        portfolio: Portfolio | None = None,
    ) -> str:
        """Optional explanation of a trade proposal."""
        return ""


BUILT_IN_AGENTS: Dict[str, Type[BaseAgent]] = {}


def _ensure_agents_loaded() -> None:
    if BUILT_IN_AGENTS:
        return
    from .alien_cipher_agent import AlienCipherAgent
    from .arbitrage import ArbitrageAgent
    from .artifact_math_agent import ArtifactMathAgent
    from .conviction import ConvictionAgent
    from .crossdex_arbitrage import CrossDEXArbitrage
    from .crossdex_rebalancer import CrossDEXRebalancer
    from .discovery import DiscoveryAgent
    from .dqn import DQNAgent
    from .emotion_agent import EmotionAgent
    from .execution import ExecutionAgent
    from .exit import ExitAgent
    from .fractal_agent import FractalAgent
    from .hedging_agent import HedgingAgent
    from .hierarchical_rl_agent import HierarchicalRLAgent
    from .memory import MemoryAgent
    from .meta_conviction import MetaConvictionAgent
    from .opportunity_cost import OpportunityCostAgent
    from .portfolio_agent import PortfolioAgent
    from .portfolio_manager import PortfolioManager
    from .portfolio_optimizer import PortfolioOptimizer
    from .ramanujan_agent import RamanujanAgent
    from .reinforcement import ReinforcementAgent
    from .rl_weight_agent import RLWeightAgent
    from .simulation import SimulationAgent
    from .smart_discovery import SmartDiscoveryAgent
    from .strange_attractor import StrangeAttractorAgent
    from .trend import TrendAgent

    BUILT_IN_AGENTS.update(
        {
            "simulation": SimulationAgent,
            "conviction": ConvictionAgent,
            "arbitrage": ArbitrageAgent,
            "exit": ExitAgent,
            "execution": ExecutionAgent,
            "memory": MemoryAgent,
            "discovery": DiscoveryAgent,
            "reinforcement": ReinforcementAgent,
            "portfolio": PortfolioAgent,
            "portfolio_manager": PortfolioManager,
            "portfolio_optimizer": PortfolioOptimizer,
            "hedging": HedgingAgent,
            "crossdex_rebalancer": CrossDEXRebalancer,
            "crossdex_arbitrage": CrossDEXArbitrage,
            "dqn": DQNAgent,
            "ppo": PPOAgent,
            "sac": SACAgent,
            "opportunity_cost": OpportunityCostAgent,
            "trend": TrendAgent,
            "smart_discovery": SmartDiscoveryAgent,
            "momentum": MomentumAgent,
            "mempool_sniper": MempoolSniperAgent,
            "mev_sandwich": MEVSandwichAgent,
            "flashloan_sandwich": FlashloanSandwichAgent,
            "meta_conviction": MetaConvictionAgent,
            "ramanujan": RamanujanAgent,
            "vanta": StrangeAttractorAgent,
            "inferna": FractalAgent,
            "alien_cipher": AlienCipherAgent,
            "artifact_math": ArtifactMathAgent,
            "rl_weight": RLWeightAgent,
            "hierarchical_rl": HierarchicalRLAgent,
            "llm_reasoner": LLMReasoner,
            "emotion": EmotionAgent,
        }
    )

    for ep in importlib.metadata.entry_points(group="solhunter_zero.agents"):
        try:
            agent_cls = ep.load()
        except Exception:  # pragma: no cover - load errors ignored
            continue
        name = getattr(agent_cls, "name", None) or ep.name
        if isinstance(name, str):
            BUILT_IN_AGENTS[name] = agent_cls


def load_agent(name: str, **kwargs) -> BaseAgent:
    """Instantiate a built-in agent by name.

    Parameters
    ----------
    name:
        The agent name. Must be one of ``BUILT_IN_AGENTS``.

    Returns
    -------
    BaseAgent
        The instantiated agent.

    Raises
    ------
    KeyError
        If ``name`` is not a known agent.
    """
    _ensure_agents_loaded()
    if name not in BUILT_IN_AGENTS:
        raise KeyError(name)
    agent_cls = BUILT_IN_AGENTS[name]
    return agent_cls(**kwargs)
