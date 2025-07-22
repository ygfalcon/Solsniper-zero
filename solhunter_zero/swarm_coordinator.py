from __future__ import annotations

from typing import Iterable, Dict

from .agents import BaseAgent
from .agents.memory import MemoryAgent


class SwarmCoordinator:
    """Compute dynamic agent weights based on historical ROI."""

    def __init__(
        self,
        memory_agent: MemoryAgent | None = None,
        base_weights: Dict[str, float] | None = None,
        regime_weights: Dict[str, Dict[str, float]] | None = None,
    ):
        self.memory_agent = memory_agent
        self.base_weights = base_weights or {}
        self.regime_weights = regime_weights or {}

    def _roi_by_agent(self, agent_names: Iterable[str]) -> Dict[str, float]:
        rois = {name: 0.0 for name in agent_names}
        if not self.memory_agent:
            return rois
        trades = self.memory_agent.memory.list_trades()
        summary: Dict[str, Dict[str, float]] = {}
        for t in trades:
            if t.reason not in rois:
                continue
            info = summary.setdefault(t.reason, {"buy": 0.0, "sell": 0.0})
            info[t.direction] += t.amount * t.price
        for name in rois:
            info = summary.get(name)
            if not info:
                continue
            spent = info.get("buy", 0.0)
            revenue = info.get("sell", 0.0)
            if spent > 0:
                rois[name] = (revenue - spent) / spent
        return rois

    def compute_weights(
        self, agents: Iterable[BaseAgent], *, regime: str | None = None
    ) -> Dict[str, float]:
        names = [a.name for a in agents]
        rois = self._roi_by_agent(names)
        base = dict(self.base_weights)
        if regime and regime in self.regime_weights:
            base.update(self.regime_weights[regime])
        if not rois:
            return {name: base.get(name, 1.0) for name in names}
        min_roi = min(rois.values())
        max_roi = max(rois.values())
        if max_roi == min_roi:
            return {name: base.get(name, 1.0) for name in names}
        weights = {}
        for name in names:
            roi = rois.get(name, 0.0)
            norm = (roi - min_roi) / (max_roi - min_roi)
            weights[name] = base.get(name, 1.0) * norm
        return weights
