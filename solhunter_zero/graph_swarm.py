from __future__ import annotations

import os
from typing import Iterable, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .swarm_coordinator import SwarmCoordinator
from .advanced_memory import AdvancedMemory
from .agents import BaseAgent
from .agents.rl_weight_agent import RLWeightAgent
from .agents.hierarchical_rl_agent import HierarchicalRLAgent


class SimpleGAT(nn.Module):
    """Minimal graph attention network for weighting agents."""

    def __init__(self, in_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.attn = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        a = torch.matmul(h, self.attn)
        a = torch.matmul(a, h.t())
        a = a.masked_fill(adj == 0, -1e9)
        a = torch.softmax(a, dim=-1)
        h = torch.matmul(a, h)
        out = self.fc2(torch.relu(h)).squeeze(-1)
        return torch.softmax(out, dim=0)


def save_model(model: SimpleGAT, path: str) -> None:
    torch.save({
        "state": model.state_dict(),
        "in_dim": model.in_dim,
        "hidden_dim": model.hidden_dim,
    }, path)


def load_model(path: str) -> SimpleGAT:
    data = torch.load(path, map_location="cpu")
    model = SimpleGAT(int(data.get("in_dim", 1)), hidden_dim=int(data.get("hidden_dim", 16)))
    model.load_state_dict(data["state"])
    model.eval()
    return model


def build_interaction_graph(memory: AdvancedMemory, agents: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return node features, adjacency matrix and ROI for ``agents``."""
    trades = memory.list_trades()
    trades.sort(key=lambda t: t.id)
    idx = {a: i for i, a in enumerate(agents)}
    counts = np.zeros(len(agents), dtype=np.float32)
    summary = {a: {"buy": 0.0, "sell": 0.0} for a in agents}
    adj = np.zeros((len(agents), len(agents)), dtype=np.float32)
    token_map: Dict[str, set[str]] = {}
    prev = None
    for t in trades:
        name = t.reason or ""
        if name not in idx:
            continue
        i = idx[name]
        counts[i] += 1
        amt = float(t.amount) * float(t.price)
        if t.direction == "buy":
            summary[name]["buy"] += amt
        else:
            summary[name]["sell"] += amt
        token_map.setdefault(t.token, set()).add(name)
        if prev is not None:
            a = prev.reason or ""
            b = name
            if a in idx and b in idx:
                adj[idx[a], idx[b]] += 1.0
        prev = t
    for tok, names in token_map.items():
        names = list(names)
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    ai = idx[names[i]]
                    aj = idx[names[j]]
                    adj[ai, aj] += 1.0
    roi = np.array([
        (summary[a]["sell"] - summary[a]["buy"]) / summary[a]["buy"] if summary[a]["buy"] > 0 else 0.0
        for a in agents
    ], dtype=np.float32)
    feats = counts.reshape(len(agents), 1)
    return feats, adj, roi


class GraphSwarm(SwarmCoordinator):
    """Coordinate agents using a graph attention network."""

    def __init__(
        self,
        memory_agent: "MemoryAgent | None" = None,
        model_path: str | None = None,
        base_weights: Dict[str, float] | None = None,
        regime_weights: Dict[str, Dict[str, float]] | None = None,
    ) -> None:
        super().__init__(memory_agent, base_weights, regime_weights)
        self.model_path = model_path
        self.model: SimpleGAT | None = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
            except Exception:
                self.model = None

    def compute_weights(
        self, agents: Iterable[BaseAgent], *, regime: str | None = None
    ) -> Dict[str, float]:
        agents = [a for a in agents if not isinstance(a, (RLWeightAgent, HierarchicalRLAgent))]
        names = [a.name for a in agents]
        if not self.model or not self.memory_agent:
            return super().compute_weights(agents, regime=regime)
        feats, adj, _roi = build_interaction_graph(self.memory_agent.memory, names)
        if feats.size == 0:
            return super().compute_weights(agents, regime=regime)
        X = torch.tensor(feats, dtype=torch.float32)
        A = torch.tensor(adj, dtype=torch.float32)
        with torch.no_grad():
            w = self.model(X, A).numpy()
        base = super().compute_weights(agents, regime=regime)
        return {n: float(w[i]) * base.get(n, 1.0) for i, n in enumerate(names)}

