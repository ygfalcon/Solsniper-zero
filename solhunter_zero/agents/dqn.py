from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
from sklearn.neural_network import MLPRegressor

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
    ) -> None:
        self.memory_agent = memory_agent or MemoryAgent()
        self.epsilon = epsilon
        self.discount = discount
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_size,),
            learning_rate_init=learning_rate,
            max_iter=200,
            random_state=0,
        )
        self.q: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
        self._fitted = False

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
            self.q[token] = [reward, -reward]
            y.append([reward, -reward])
        if not X:
            return
        X_arr = np.array(X)
        y_arr = np.array(y)
        self.model.fit(X_arr, y_arr)
        self._fitted = True

    # ------------------------------------------------------------------
    async def propose_trade(self, token: str, portfolio: Portfolio) -> List[Dict[str, Any]]:
        self.train(portfolio)
        state = np.array([self._state(token, portfolio)])
        if self._fitted:
            _ = self.model.predict(state)
        q = self.q[token]
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
