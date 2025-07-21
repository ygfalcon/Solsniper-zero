from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Dict, Any

from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from . import BaseAgent
from .memory import MemoryAgent
from ..portfolio import Portfolio


class DQNAgent(BaseAgent):
    """Deep Q-Network agent that learns from trade history."""

    name = "dqn"

    def __init__(
        self,
        memory_agent: MemoryAgent | None = None,
        *,
        hidden_size: int = 8,
        learning_rate: float = 0.001,
        epsilon: float = 0.1,
        discount: float = 0.95,
        model_path: str | Path = "dqn_model.pt",
    ) -> None:
        self.memory_agent = memory_agent or MemoryAgent()
        self.epsilon = epsilon
        self.discount = discount
        self.model_path = Path(model_path)

        self.model = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self._fitted = False

        if self.model_path.exists():
            data = torch.load(self.model_path)
            self.model.load_state_dict(data.get("model_state", {}))
            opt_state = data.get("optimizer_state")
            if opt_state:
                self.optimizer.load_state_dict(opt_state)
            self._fitted = True

    # ------------------------------------------------------------------
    def _state(self, token: str, portfolio: Portfolio) -> List[float]:
        pos = portfolio.balances.get(token)
        amt = float(pos.amount) if pos else 0.0
        return [amt]

    def train(self, portfolio: Portfolio) -> None:
        trades = self.memory_agent.memory.list_trades()
        profits: Dict[str, float] = defaultdict(float)
        for t in trades:
            value = float(t.amount) * float(t.price)
            if t.direction == "buy":
                profits[t.token] -= value
            else:
                profits[t.token] += value
        X: List[List[float]] = []
        y: List[List[float]] = []
        for token, reward in profits.items():
            state = self._state(token, portfolio)
            X.append(state)
            y.append([reward, -reward])
        if not X:
            return
        X_arr = torch.tensor(np.array(X), dtype=torch.float32)
        y_arr = torch.tensor(np.array(y), dtype=torch.float32)

        self.model.train()
        for _ in range(100):
            self.optimizer.zero_grad()
            pred = self.model(X_arr)
            loss = self.loss_fn(pred, y_arr)
            loss.backward()
            self.optimizer.step()
        self._fitted = True

        if self.model_path:
            torch.save(
                {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                },
                self.model_path,
            )

    # ------------------------------------------------------------------
    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        self.train(portfolio)
        state = torch.tensor([self._state(token, portfolio)], dtype=torch.float32)
        if self._fitted:
            with torch.no_grad():
                q = self.model(state)[0].numpy()
        else:
            q = [0.0, 0.0]
        if random.random() < self.epsilon:
            action = random.choice(["buy", "sell"])
        else:
            action = "buy" if q[0] >= q[1] else "sell"
        if action == "buy":
            return [{"token": token, "side": "buy", "amount": 1.0, "price": 0.0, "agent": self.name}]
        position = portfolio.balances.get(token)
        if position:
            return [{"token": token, "side": "sell", "amount": position.amount, "price": 0.0, "agent": self.name}]
        return []
