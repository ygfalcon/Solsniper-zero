from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn, optim
import torch.nn.functional as F

import asyncio
import logging

from . import BaseAgent
from .memory import MemoryAgent
from ..offline_data import OfflineData
from ..order_book_ws import snapshot
from ..portfolio import Portfolio
from ..replay import ReplayBuffer
from .. import models
from ..simulation import fetch_token_metrics


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
        price_model_path: str | None = None,
    ) -> None:
        self.memory_agent = memory_agent or MemoryAgent()
        self.offline_data = OfflineData(data_url)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = int(epochs)
        self.model_path = Path(model_path)
        self.replay = ReplayBuffer(replay_url)
        self.price_model_path = price_model_path or os.getenv("PRICE_MODEL_PATH")
        self._seen_ids: set[int] = set()
        self._task: asyncio.Task | None = None
        self._logger = logging.getLogger(__name__)

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
        depth, imb, _ = snapshot(token)
        return [amt, depth, imb]

    def _predict_return(self, token: str) -> float:
        if not self.price_model_path:
            return 0.0
        model = models.get_model(self.price_model_path, reload=True)
        if not model:
            return 0.0
        metrics = fetch_token_metrics(token)
        ph = metrics.get("price_history") or []
        lh = metrics.get("liquidity_history") or []
        dh = metrics.get("depth_history") or []
        sh = metrics.get("slippage_history") or []
        vh = metrics.get("volume_history") or []
        th = metrics.get("tx_count_history") or []
        n = min(len(ph), len(lh), len(dh), len(sh or ph), len(vh or ph), len(th or ph))
        if n < 30:
            return 0.0
        seq = np.column_stack([
            ph[-30:],
            lh[-30:],
            dh[-30:],
            (sh or [0] * n)[-30:],
            (vh or [0] * n)[-30:],
            (th or [0] * n)[-30:],
        ])
        try:
            return float(model.predict(seq))
        except Exception:
            return 0.0

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

    async def _online_loop(self, interval: float = 60.0) -> None:
        """Continuously train the agent from new replay data."""
        while True:
            try:
                self.train()
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
        self.train()
        state = torch.tensor([self._state(token, portfolio)], dtype=torch.float32)
        with torch.no_grad():
            logits = self.actor(state)[0]
        pred = self._predict_return(token)
        logits = logits + torch.tensor([pred, -pred])
        action = "buy" if logits[0] >= logits[1] else "sell"
        if action == "buy":
            return [{"token": token, "side": "buy", "amount": 1.0, "price": 0.0, "agent": self.name}]
        position = portfolio.balances.get(token)
        if position:
            return [{"token": token, "side": "sell", "amount": position.amount, "price": 0.0, "agent": self.name}]
        return []
