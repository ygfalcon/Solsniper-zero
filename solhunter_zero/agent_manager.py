from __future__ import annotations

import asyncio
import json
import os
from typing import Iterable, Dict, Any, List

import tomllib

from .agents import BaseAgent, load_agent
from .agents.execution import ExecutionAgent
from .agents.swarm import AgentSwarm
from .agents.memory import MemoryAgent
from .agents.emotion_agent import EmotionAgent
from .agents.discovery import DiscoveryAgent
from .swarm_coordinator import SwarmCoordinator


class StrategySelector:
    """Rank agents by recent ROI logged by ``MemoryAgent``."""

    def __init__(self, memory_agent: MemoryAgent, *, vote_threshold: float = 0.0) -> None:
        self.memory_agent = memory_agent
        self.vote_threshold = float(vote_threshold)

    def _roi_by_agent(self, names: Iterable[str]) -> Dict[str, float]:
        rois = {n: 0.0 for n in names}
        trades = self.memory_agent.memory.list_trades() if self.memory_agent else []
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

    def rank_agents(self, agents: Iterable[BaseAgent]) -> List[str]:
        names = [a.name for a in agents]
        rois = self._roi_by_agent(names)
        ranked = sorted(names, key=lambda n: rois.get(n, 0.0), reverse=True)
        return ranked

    def weight_agents(
        self, agents: Iterable[BaseAgent], base_weights: Dict[str, float]
    ) -> tuple[List[BaseAgent], Dict[str, float]]:
        names = [a.name for a in agents]
        rois = self._roi_by_agent(names)
        if not rois:
            return list(agents), dict(base_weights)

        ranked = sorted(names, key=lambda n: rois.get(n, 0.0), reverse=True)
        top = ranked[0]
        second_roi = rois.get(ranked[1], 0.0) if len(ranked) > 1 else float("-inf")
        if rois[top] - second_roi >= self.vote_threshold:
            selected = [a for a in agents if a.name == top]
            return selected, {top: base_weights.get(top, 1.0)}

        max_roi = max(rois.values())
        min_roi = min(rois.values())
        if max_roi == min_roi:
            return list(agents), dict(base_weights)

        weights = {}
        for name in names:
            roi = rois.get(name, 0.0)
            norm = (roi - min_roi) / (max_roi - min_roi)
            weights[name] = base_weights.get(name, 1.0) * (1.0 + norm)
        return list(agents), weights



class AgentManager:
    """Manage and coordinate trading agents and execute actions."""

    def __init__(
        self,
        agents: Iterable[BaseAgent],
        executor: ExecutionAgent | None = None,
        *,
        weights: Dict[str, float] | None = None,
        memory_agent: MemoryAgent | None = None,
        emotion_agent: EmotionAgent | None = None,
        weights_path: str | os.PathLike | None = None,
        strategy_selection: bool = False,
        vote_threshold: float = 0.0,
    ):
        self.agents = list(agents)
        self.executor = executor or ExecutionAgent()
        self.weights_path = (
            str(weights_path) if weights_path is not None else "weights.json"
        )

        file_weights: Dict[str, float] = {}
        if self.weights_path and os.path.exists(self.weights_path):
            file_weights = self._load_weights(self.weights_path)

        init_weights = weights or {}
        self.weights = {**file_weights, **init_weights}

        self.memory_agent = memory_agent or next(
            (a for a in self.agents if isinstance(a, MemoryAgent)),
            None,
        )

        self.emotion_agent = emotion_agent or next(
            (a for a in self.agents if isinstance(a, EmotionAgent)),
            None,
        )

        self.coordinator = SwarmCoordinator(self.memory_agent, self.weights)

        self.strategy_selection = strategy_selection
        self.vote_threshold = float(vote_threshold)
        self.selector: StrategySelector | None = None
        if self.strategy_selection and self.memory_agent:
            self.selector = StrategySelector(
                self.memory_agent, vote_threshold=self.vote_threshold
            )

    async def evaluate(self, token: str, portfolio) -> List[Dict[str, Any]]:
        agents = list(self.agents)
        weights = self.coordinator.compute_weights(agents)
        if self.selector:
            agents, weights = self.selector.weight_agents(agents, weights)
        swarm = AgentSwarm(agents)
        return await swarm.propose(token, portfolio, weights=weights)


    async def execute(self, token: str, portfolio) -> List[Any]:
        actions = await self.evaluate(token, portfolio)
        results = []
        for action in actions:
            result = await self.executor.execute(action)
            if self.emotion_agent:
                emotion = self.emotion_agent.evaluate(action, result)
                action["emotion"] = emotion
            results.append(result)
            if self.memory_agent:
                await self.memory_agent.log(action)
        return results

    def update_weights(self) -> None:
        """Adjust agent weights based on historical trade ROI."""
        if not self.memory_agent:
            return

        trades = self.memory_agent.memory.list_trades()
        summary: Dict[str, Dict[str, float]] = {}
        for t in trades:
            name = t.reason or ""
            info = summary.setdefault(name, {"buy": 0.0, "sell": 0.0})
            info[t.direction] += t.amount * t.price

        for name, info in summary.items():
            spent = info.get("buy", 0.0)
            revenue = info.get("sell", 0.0)
            if spent <= 0:
                continue
            roi = (revenue - spent) / spent
            if roi > 0:
                self.weights[name] = self.weights.get(name, 1.0) * 1.1
            elif roi < 0:
                self.weights[name] = self.weights.get(name, 1.0) * 0.9

        self.coordinator.base_weights = self.weights

    # ------------------------------------------------------------------
    #  Persistence helpers
    # ------------------------------------------------------------------
    def _load_weights(self, path: str | os.PathLike) -> Dict[str, float]:
        try:
            if str(path).endswith(".toml"):
                with open(path, "rb") as fh:
                    data = tomllib.load(fh)
            else:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): float(v) for k, v in data.items()}

    def save_weights(self, path: str | os.PathLike | None = None) -> None:
        path = path or self.weights_path
        if not path:
            return
        if str(path).endswith(".toml"):
            lines = [f"{k} = {v}" for k, v in self.weights.items()]
            content = "\n".join(lines) + "\n"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
        else:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.weights, fh)

    # ------------------------------------------------------------------
    #  Convenience helpers
    # ------------------------------------------------------------------
    async def discover_tokens(
        self,
        *,
        offline: bool = False,
        token_file: str | None = None,
        method: str | None = None,
    ) -> List[str]:
        for agent in self.agents:
            if isinstance(agent, DiscoveryAgent):
                return await agent.discover_tokens(
                    offline=offline, token_file=token_file, method=method
                )
        disc = DiscoveryAgent()
        return await disc.discover_tokens(
            offline=offline, token_file=token_file, method=method
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "AgentManager | None":
        names = cfg.get("agents", [])
        if isinstance(names, str):
            try:
                import ast

                parsed = ast.literal_eval(names)
                if isinstance(parsed, list):
                    names = parsed
                else:
                    names = [n.strip() for n in names.split(",") if n.strip()]
            except Exception:
                names = [n.strip() for n in names.split(",") if n.strip()]
        agents = []
        for name in names:
            try:
                agents.append(load_agent(name))
            except KeyError:
                continue
        weights = cfg.get("agent_weights") or {}
        if isinstance(weights, str):
            try:
                import ast

                parsed_w = ast.literal_eval(weights)
                if isinstance(parsed_w, dict):
                    weights = parsed_w
                else:
                    weights = {}
            except Exception:
                weights = {}
        weights_path = cfg.get("weights_path")
        memory_agent = next(
            (a for a in agents if isinstance(a, MemoryAgent)),
            None,
        )
        emotion_agent = next(
            (a for a in agents if isinstance(a, EmotionAgent)),
            None,
        )
        strategy_selection = bool(cfg.get("strategy_selection", False))
        vote_threshold = float(cfg.get("vote_threshold", 0.0) or 0.0)
        if not agents:
            return None
        return cls(
            agents,
            weights=weights,
            memory_agent=memory_agent,
            emotion_agent=emotion_agent,
            weights_path=weights_path,
            strategy_selection=strategy_selection,
            vote_threshold=vote_threshold,
        )

