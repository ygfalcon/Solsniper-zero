from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type

from ..portfolio import Portfolio
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Imports for type checking only to avoid circular imports
    from .simulation import SimulationAgent
    from .conviction import ConvictionAgent
    from .arbitrage import ArbitrageAgent
    from .exit import ExitAgent
    from .execution import ExecutionAgent
    from .memory import MemoryAgent
    from .discovery import DiscoveryAgent
    from .reinforcement import ReinforcementAgent


class BaseAgent(ABC):
    """Abstract trading agent."""

    name: str = "base"

    @abstractmethod
    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        """Return proposed trade actions for ``token``."""
        raise NotImplementedError


BUILT_IN_AGENTS: Dict[str, Type[BaseAgent]] = {}


def _ensure_agents_loaded() -> None:
    if BUILT_IN_AGENTS:
        return
    from .simulation import SimulationAgent
    from .conviction import ConvictionAgent
    from .arbitrage import ArbitrageAgent
    from .exit import ExitAgent
    from .execution import ExecutionAgent
    from .memory import MemoryAgent
    from .discovery import DiscoveryAgent

    BUILT_IN_AGENTS.update({
        "simulation": SimulationAgent,
        "conviction": ConvictionAgent,
        "arbitrage": ArbitrageAgent,
        "exit": ExitAgent,
        "execution": ExecutionAgent,
        "memory": MemoryAgent,
        "discovery": DiscoveryAgent,
        "reinforcement": ReinforcementAgent,
    })


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
