from __future__ import annotations

import random
from typing import List, Dict, Any

import asyncio
import logging

from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from . import BaseAgent
from .memory import MemoryAgent
from ..portfolio import Portfolio
from ..replay import ReplayBuffer


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
        replay_url: str = "sqlite:///replay.db",
    ) -> None:
        self.memory_agent = memory_agent or MemoryAgent()
        self.epsilon = epsilon
        self.discount = discount
        self.model_path = Path(model_path)
        self.replay = ReplayBuffer(replay_url)
        self._seen_ids: set[int] = set()
        self._task: asyncio.Task | None = None
        self._logger = logging.getLogger(__name__)

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
        for t in trades:
            tid = getattr(t, "id", None)
            if tid is not None and tid in self._seen_ids:
                continue
            if tid is not None:
                self._seen_ids.add(tid)
            reward = float(t.amount) * float(t.price)
            if t.direction == "buy":
                reward = -reward
            emotion = getattr(t, "emotion", "")
            if emotion == "regret":
                continue
            self.replay.add([float(t.amount)], t.direction, reward, emotion)

        batch = self.replay.sample(32)
        if not batch:
            return

        X: List[List[float]] = []
        y: List[List[float]] = []
        for state, action, reward, _ in batch:
            X.append(list(state))
            y.append([reward, -reward])

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

    async def _online_loop(self, interval: float = 60.0) -> None:
        """Continuously train the agent from new replay data."""
        portfolio = Portfolio()
        while True:
            try:
                self.train(portfolio)
            except Exception as exc:  # pragma: no cover - logging
                self._logger.error("online train failed: %s", exc)
            await asyncio.sleep(interval)

    def start_online_learning(self, interval: float = 60.0) -> None:
        """Start background task that updates the model periodically."""
        if self._task is None:
            self._task = asyncio.create_task(self._online_loop(interval))

    # ------------------------------------------------------------------
    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        self.start_online_learning()
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
