from __future__ import annotations

import json
import os
import random
from typing import Any, List

from .agents.memory import MemoryAgent


class PopulationRL:
    """Explore agent weights and risk parameters using a simple evolutionary loop."""

    def __init__(
        self,
        memory_agent: MemoryAgent | None = None,
        *,
        population_size: int = 4,
        weights_path: str = "weights.json",
    ) -> None:
        self.memory_agent = memory_agent
        self.population_size = int(population_size)
        self.weights_path = weights_path
        self.population: List[dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.weights_path and os.path.exists(self.weights_path):
            try:
                with open(self.weights_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    self.population = data
                elif isinstance(data, dict):
                    self.population = [data]
            except Exception:
                self.population = []
        if not self.population:
            self.population = [
                {
                    "weights": {},
                    "risk": {"risk_multiplier": 1.0},
                    "score": 0.0,
                }
            ]

    def _save(self) -> None:
        if not self.weights_path:
            return
        with open(self.weights_path, "w", encoding="utf-8") as fh:
            json.dump(self.population, fh)

    # ------------------------------------------------------------------
    def _roi(self, agent: str) -> float:
        if not self.memory_agent:
            return 0.0
        trades = self.memory_agent.memory.list_trades()
        spent = revenue = 0.0
        for t in trades:
            if t.reason != agent:
                continue
            val = float(t.amount) * float(t.price)
            if t.direction == "buy":
                spent += val
            else:
                revenue += val
        return (revenue - spent) / spent if spent > 0 else 0.0

    def _score_cfg(self, cfg: dict[str, Any]) -> float:
        weights = cfg.get("weights", {})
        risk = cfg.get("risk", {})
        score = 0.0
        for name, w in weights.items():
            score += float(w) * self._roi(name)
        risk_mult = float(risk.get("risk_multiplier", 1.0))
        return score * risk_mult

    # ------------------------------------------------------------------
    def evolve(self) -> dict[str, Any]:
        """Generate a new population and persist the best configuration."""
        for cfg in self.population:
            cfg["score"] = self._score_cfg(cfg)
        self.population.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        keep = self.population[: max(1, len(self.population) // 2)]
        while len(keep) < self.population_size:
            parent = random.choice(keep)
            child = {
                "weights": {
                    k: max(0.0, float(v) * random.uniform(0.8, 1.2))
                    for k, v in parent.get("weights", {}).items()
                },
                "risk": {
                    k: max(0.0, float(v) * random.uniform(0.8, 1.2))
                    for k, v in parent.get("risk", {}).items()
                },
                "score": 0.0,
            }
            keep.append(child)
        self.population = keep
        self._save()
        return self.population[0]

    # ------------------------------------------------------------------
    def best_config(self) -> dict[str, Any]:
        self.population.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return self.population[0]


if __name__ == "__main__":  # pragma: no cover - simple CLI
    import argparse
    from .memory import Memory

    ap = argparse.ArgumentParser(description="Evolve RL population weights")
    ap.add_argument("--memory", default="sqlite:///memory.db")
    ap.add_argument("--weights", dest="weights_path", default="weights.json")
    ap.add_argument("--population-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    if args.num_workers is not None:
        os.environ["RL_NUM_WORKERS"] = str(args.num_workers)

    mgr = MemoryAgent(Memory(args.memory))
    rl = PopulationRL(mgr, population_size=args.population_size, weights_path=args.weights_path)
    best = rl.evolve()
    print(json.dumps(best))

