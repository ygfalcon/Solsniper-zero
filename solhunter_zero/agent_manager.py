from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Iterable, Dict, Any, List

from .backtester import backtest_weighted, DEFAULT_STRATEGIES
from .backtest_cli import bayesian_optimize_weights
from .advanced_memory import AdvancedMemory

import logging
import tomllib

from .agents import BaseAgent, load_agent
from .agents.execution import ExecutionAgent
from .execution import EventExecutor
from .agents.swarm import AgentSwarm
from .agents.memory import MemoryAgent
from .agents.emotion_agent import EmotionAgent
from .agents.discovery import DiscoveryAgent
from .swarm_coordinator import SwarmCoordinator
from .regime import detect_regime
from . import mutation
from .event_bus import publish, subscription
from .multi_rl import PopulationRL


logger = logging.getLogger(__name__)


class StrategySelector:
    """Rank agents by recent ROI logged by ``MemoryAgent``."""

    def __init__(
        self, memory_agent: MemoryAgent, *, vote_threshold: float = 0.0
    ) -> None:
        self.memory_agent = memory_agent
        self.vote_threshold = float(vote_threshold)

    def _roi_by_agent(self, names: Iterable[str]) -> Dict[str, float]:
        rois = {n: 0.0 for n in names}
        trades = (
            self.memory_agent.memory.list_trades() if self.memory_agent else []
        )
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
        second_roi = (
            rois.get(ranked[1], 0.0) if len(ranked) > 1 else float("-inf")
        )
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
        population_rl: PopulationRL | None = None,
        weights_path: str | os.PathLike | None = None,
        strategy_selection: bool = False,
        vote_threshold: float = 0.0,
        mutation_path: str | os.PathLike | None = "mutation_state.json",
        depth_service: bool = False,
        priority_rpc: list[str] | None = None,
        regime_weights: Dict[str, Dict[str, float]] | None = None,
        evolve_interval: int = 1,
        mutation_threshold: float = 0.0,
    ):
        self.agents = list(agents)
        self.executor = executor or ExecutionAgent(
            depth_service=depth_service,
            priority_rpc=priority_rpc,
        )
        self.weights_path = (
            str(weights_path) if weights_path is not None else "weights.json"
        )
        file_weights: Dict[str, float] = {}
        if self.weights_path and os.path.exists(self.weights_path):
            file_weights = self._load_weights(self.weights_path)

        init_weights = weights or {}
        self.weights = {**file_weights, **init_weights}
        self.regime_weights = regime_weights or {}

        self.memory_agent = memory_agent or next(
            (a for a in self.agents if isinstance(a, MemoryAgent)),
            None,
        )

        self.emotion_agent = emotion_agent or next(
            (a for a in self.agents if isinstance(a, EmotionAgent)),
            None,
        )

        self.population_rl = population_rl

        self.coordinator = SwarmCoordinator(
            self.memory_agent, self.weights, self.regime_weights
        )

        self.mutation_path = (
            str(mutation_path)
            if mutation_path is not None
            else "mutation_state.json"
        )
        self.mutation_state = mutation.load_state(self.mutation_path)

        self.strategy_selection = strategy_selection
        self.vote_threshold = float(vote_threshold)
        self.selector: StrategySelector | None = None
        if self.strategy_selection and self.memory_agent:
            self.selector = StrategySelector(
                self.memory_agent, vote_threshold=self.vote_threshold
            )

        self.evolve_interval = int(evolve_interval)
        self.mutation_threshold = float(mutation_threshold)

        self.depth_service = depth_service
        self._event_executors: Dict[str, EventExecutor] = {}
        self._event_tasks: Dict[str, asyncio.Task] = {}
        self._subscriptions: list[Any] = []

        sub = subscription("config_updated", self._update_from_config)
        sub.__enter__()
        self._subscriptions.append(sub)

    async def evaluate(self, token: str, portfolio) -> List[Dict[str, Any]]:
        agents = list(self.agents)
        regime = detect_regime(portfolio.price_history.get(token, []))
        weights = self.coordinator.compute_weights(agents, regime=regime)
        if self.selector:
            agents, weights = self.selector.weight_agents(agents, weights)
        swarm = AgentSwarm(agents)
        return await swarm.propose(token, portfolio, weights=weights)

    async def execute(self, token: str, portfolio) -> List[Any]:
        actions = await self.evaluate(token, portfolio)
        results = []
        if self.depth_service and token not in self._event_executors:
            execer = EventExecutor(
                token,
                priority_rpc=getattr(self.executor, "priority_rpc", None),
            )
            self._event_executors[token] = execer
            self.executor.add_executor(token, execer)
            self._event_tasks[token] = asyncio.create_task(execer.run())
        for action in actions:
            result = await self.executor.execute(action)
            if self.emotion_agent:
                emotion = self.emotion_agent.evaluate(action, result)
                action["emotion"] = emotion
            results.append(result)
            if self.memory_agent:
                await self.memory_agent.log(action)
            publish("action_executed", {"action": action, "result": result})
        return results

    def _update_from_config(self, cfg: dict) -> None:
        weights = cfg.get("agent_weights")
        if isinstance(weights, str):
            try:
                import ast

                parsed = ast.literal_eval(weights)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                weights = parsed
            else:
                weights = {}
        if isinstance(weights, dict):
            for k, v in weights.items():
                try:
                    self.weights[str(k)] = float(v)
                except Exception:
                    continue
            self.coordinator.base_weights = self.weights

        ei = cfg.get("evolve_interval")
        if ei is not None:
            try:
                self.evolve_interval = int(ei)
            except Exception:
                pass
        mt = cfg.get("mutation_threshold")
        if mt is not None:
            try:
                self.mutation_threshold = float(mt)
            except Exception:
                pass

        reg_w = cfg.get("regime_weights")
        if reg_w is not None:
            if isinstance(reg_w, str):
                try:
                    import ast

                    parsed_r = ast.literal_eval(reg_w)
                    if isinstance(parsed_r, dict):
                        reg_w = parsed_r
                except Exception:
                    reg_w = None
            if isinstance(reg_w, dict):
                self.regime_weights = {
                    str(k): {str(sk): float(sv) for sk, sv in v.items()}
                    for k, v in reg_w.items()
                }
                self.coordinator.regime_weights = self.regime_weights

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

        adv_mem = None
        if isinstance(self.memory_agent.memory, AdvancedMemory):
            adv_mem = self.memory_agent.memory

        for name, info in summary.items():
            spent = info.get("buy", 0.0)
            revenue = info.get("sell", 0.0)
            if spent <= 0:
                continue
            roi = (revenue - spent) / spent
            success_rate = 0.0
            if adv_mem is not None:
                try:
                    success_rate = adv_mem.simulation_success_rate(name)
                except Exception:
                    success_rate = 0.0
            factor = 1.0
            if roi > 0:
                factor = 1.1
            elif roi < 0:
                factor = 0.9
            else:
                continue
            factor *= 1.0 + success_rate
            self.weights[name] = self.weights.get(name, 1.0) * factor

        self.coordinator.base_weights = self.weights
        self.save_weights()
        publish("weights_updated", dict(self.weights))

    # ------------------------------------------------------------------
    #  Mutation helpers
    # ------------------------------------------------------------------
    def _roi_by_agent(self, names: Iterable[str]) -> Dict[str, float]:
        if not self.memory_agent:
            return {n: 0.0 for n in names}
        trades = self.memory_agent.memory.list_trades()
        summary: Dict[str, Dict[str, float]] = {}
        for t in trades:
            if t.reason not in names:
                continue
            info = summary.setdefault(t.reason, {"buy": 0.0, "sell": 0.0})
            info[t.direction] += float(t.amount) * float(t.price)
        rois = {n: 0.0 for n in names}
        for name, info in summary.items():
            spent = info.get("buy", 0.0)
            revenue = info.get("sell", 0.0)
            if spent > 0:
                rois[name] = (revenue - spent) / spent
        return rois

    def spawn_mutations(self, count: int = 1) -> List[BaseAgent]:
        base_agents = [
            a
            for a in self.agents
            if a.name not in self.mutation_state.get("active", [])
            and not isinstance(a, MemoryAgent)
        ]
        spawned: List[BaseAgent] = []
        if not base_agents:
            return spawned
        for _ in range(count):
            base = random.choice(base_agents)
            name = (
                f"{base.name}_m"
                f"{len(self.mutation_state.get('active', [])) + 1}"
            )
            mutated = mutation.mutate_agent(base, name=name)
            self.agents.append(mutated)
            self.mutation_state.setdefault("active", []).append(mutated.name)
            spawned.append(mutated)
        return spawned

    def prune_underperforming(self, threshold: float = 0.0) -> None:
        active = list(self.mutation_state.get("active", []))
        if not active:
            return
        rois = self._roi_by_agent(active)
        keep = []
        remaining_agents = []
        for agent in self.agents:
            if agent.name in active:
                if rois.get(agent.name, 0.0) >= threshold:
                    remaining_agents.append(agent)
                    keep.append(agent.name)
            else:
                remaining_agents.append(agent)
        self.agents = remaining_agents
        self.mutation_state["active"] = keep
        self.mutation_state.setdefault("roi", {}).update(rois)

    def save_mutation_state(
        self, path: str | os.PathLike | None = None
    ) -> None:
        path = path or self.mutation_path
        if path:
            mutation.save_state(self.mutation_state, str(path))

    def _load_price_history(self) -> List[float]:
        path = os.path.join(
            os.path.dirname(__file__), "..", "datasets", "sample_ticks.json"
        )
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data and isinstance(data[0], dict):
                return [float(d.get("price", 0.0)) for d in data]
            return [float(x) for x in data]
        except Exception:
            return []

    def evolve(self, spawn_count: int = 1, threshold: float = 0.0) -> None:
        if self.population_rl is not None:
            best = self.population_rl.evolve()
            best_weights = best.get("weights", {}) if isinstance(best, dict) else {}
            if isinstance(best_weights, dict):
                self.weights.update(best_weights)
                self.coordinator.base_weights = self.weights
                self.save_weights()

        new_agents = self.spawn_mutations(spawn_count)
        prices = self._load_price_history()
        if prices:
            baseline = backtest_weighted(
                prices, self.weights, strategies=DEFAULT_STRATEGIES
            ).roi
            for agent in list(new_agents):
                test_weights = dict(self.weights)
                test_weights[agent.name] = 1.0
                roi = backtest_weighted(
                    prices, test_weights, strategies=DEFAULT_STRATEGIES
                ).roi
                if roi < baseline:
                    self.agents.remove(agent)
                    self.mutation_state.get("active", []).remove(agent.name)
                else:
                    if self.memory_agent and isinstance(
                        self.memory_agent.memory, AdvancedMemory
                    ):
                        sim_id = self.memory_agent.memory.log_simulation(
                            "MUT",
                            expected_roi=roi,
                            success_prob=1.0 if roi > 0 else 0.0,
                        )
                        self.memory_agent.memory.log_trade(
                            token="MUT",
                            direction="buy",
                            amount=1.0,
                            price=1.0,
                            reason=agent.name,
                            simulation_id=sim_id,
                        )
                        self.memory_agent.memory.log_trade(
                            token="MUT",
                            direction="sell",
                            amount=1.0,
                            price=1.0 + roi,
                            reason=agent.name,
                            simulation_id=sim_id,
                        )

            keys = [name for name, _ in DEFAULT_STRATEGIES]
            try:
                opt = bayesian_optimize_weights(
                    prices, keys, DEFAULT_STRATEGIES, iterations=10
                )
                self.weights.update(opt)
                self.coordinator.base_weights = self.weights
            except Exception as exc:
                logger.exception(
                    "bayesian_optimize_weights failed", exc_info=exc
                )

        self.prune_underperforming(threshold)
        if self.mutation_path:
            self.save_mutation_state()

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
        depth_service = bool(cfg.get("depth_service", False))
        priority_rpc = cfg.get("priority_rpc")
        if isinstance(priority_rpc, str):
            priority_rpc = [
                u.strip() for u in priority_rpc.split(",") if u.strip()
            ]
        elif not isinstance(priority_rpc, list):
            priority_rpc = None
        regime_weights = cfg.get("regime_weights") or {}
        if isinstance(regime_weights, str):
            try:
                import ast

                parsed_r = ast.literal_eval(regime_weights)
                if isinstance(parsed_r, dict):
                    regime_weights = parsed_r
                else:
                    regime_weights = {}
            except Exception:
                regime_weights = {}
        evolve_interval = int(cfg.get("evolve_interval", 1))
        mutation_threshold = float(cfg.get("mutation_threshold", 0.0))

        jito_rpc_url = cfg.get("jito_rpc_url")
        jito_auth = cfg.get("jito_auth")
        jito_ws_url = cfg.get("jito_ws_url")
        jito_ws_auth = cfg.get("jito_ws_auth")
        if jito_rpc_url and os.getenv("JITO_RPC_URL") is None:
            os.environ["JITO_RPC_URL"] = str(jito_rpc_url)
        if jito_auth and os.getenv("JITO_AUTH") is None:
            os.environ["JITO_AUTH"] = str(jito_auth)
        if jito_ws_url and os.getenv("JITO_WS_URL") is None:
            os.environ["JITO_WS_URL"] = str(jito_ws_url)
        if jito_ws_auth and os.getenv("JITO_WS_AUTH") is None:
            os.environ["JITO_WS_AUTH"] = str(jito_ws_auth)

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
            depth_service=depth_service,
            priority_rpc=priority_rpc,
            regime_weights=regime_weights,
            evolve_interval=evolve_interval,
            mutation_threshold=mutation_threshold,
        )

    def close(self) -> None:
        for sub in getattr(self, "_subscriptions", []):
            try:
                sub.__exit__(None, None, None)
            except Exception:
                pass
