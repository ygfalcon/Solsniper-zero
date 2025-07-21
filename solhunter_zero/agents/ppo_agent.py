from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn, optim
import torch.nn.functional as F

from . import BaseAgent
from .memory import MemoryAgent
from ..offline_data import OfflineData
from ..order_book_ws import snapshot
from ..portfolio import Portfolio
from ..replay import ReplayBuffer


class PPOAgent(BaseAgent):
    """Actor-critic agent trained with PPO."""

    name = "ppo"

    def __init__(
        self,
        memory_agent: MemoryAgent | None = None,
        *,
        data_url: str = "sqlite:///offline_data.db",
        hidden_size: int = 32,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        epochs: int = 5,
        model_path: str | Path = "ppo_model.pt",
        replay_url: str = "sqlite:///replay.db",
    ) -> None:
        self.memory_agent = memory_agent or MemoryAgent()
        self.offline_data = OfflineData(data_url)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = int(epochs)
        self.model_path = Path(model_path)
        self.replay = ReplayBuffer(replay_url)
        self._seen_ids: set[int] = set()

        self.actor = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
        )
        self._fitted = False

        if self.model_path.exists():
            data = torch.load(self.model_path)
            self.actor.load_state_dict(data.get("actor_state", {}))
            self.critic.load_state_dict(data.get("critic_state", {}))
            opt_state = data.get("optim_state")
            if opt_state:
                self.optimizer.load_state_dict(opt_state)
            self._fitted = True

    # ------------------------------------------------------------------
    def _state(self, token: str, portfolio: Portfolio) -> List[float]:
        pos = portfolio.balances.get(token)
        amt = float(pos.amount) if pos else 0.0
        depth, imb = snapshot(token)
        return [amt, depth, imb]

    def _log_trades(self) -> None:
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
            state = [float(t.amount), float(t.price), 0.0]
            self.replay.add(state, t.direction, reward, getattr(t, "emotion", ""))

    def train(self) -> None:
        self._log_trades()
        batch = self.replay.sample(64)
        if not batch:
            return

        states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        actions = torch.tensor([0 if b[1] == "buy" else 1 for b in batch])
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)

        with torch.no_grad():
            old_log_probs = (
                torch.distributions.Categorical(logits=self.actor(states))
                .log_prob(actions)
            )
            values = self.critic(states).squeeze()
            advantages = rewards - values
            returns = rewards

        for _ in range(self.epochs):
            dist = torch.distributions.Categorical(logits=self.actor(states))
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            s1 = ratio * advantages
            s2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(s1, s2).mean()

            value_pred = self.critic(states).squeeze()
            critic_loss = F.mse_loss(value_pred, returns)

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self._fitted = True
        torch.save(
            {
                "actor_state": self.actor.state_dict(),
                "critic_state": self.critic.state_dict(),
                "optim_state": self.optimizer.state_dict(),
            },
            self.model_path,
        )

    # ------------------------------------------------------------------
    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        self.train()
        state = torch.tensor([self._state(token, portfolio)], dtype=torch.float32)
        with torch.no_grad():
            logits = self.actor(state)[0]
        action = "buy" if logits[0] >= logits[1] else "sell"
        if action == "buy":
            return [{"token": token, "side": "buy", "amount": 1.0, "price": 0.0, "agent": self.name}]
        position = portfolio.balances.get(token)
        if position:
            return [{"token": token, "side": "sell", "amount": position.amount, "price": 0.0, "agent": self.name}]
        return []
