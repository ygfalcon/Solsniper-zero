from __future__ import annotations

import os
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn, optim

from . import BaseAgent
from .memory import MemoryAgent
from ..offline_data import OfflineData
from ..order_book_ws import snapshot
from ..portfolio import Portfolio
from ..replay import ReplayBuffer
from ..regime import detect_regime


class TransformerActorCritic(nn.Module):
    """Actor-critic network using a transformer encoder."""

    def __init__(self, state_dim: int, hidden_size: int = 32, nhead: int = 2, num_layers: int = 1) -> None:
        super().__init__()
        self.embed = nn.Linear(state_dim, hidden_size)
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.actor = nn.Linear(hidden_size, 2)
        self.critic = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(x).unsqueeze(0)
        x = self.encoder(x).squeeze(0)
        return self.actor(x), self.critic(x).squeeze(-1)


class TransformerRLAgent(BaseAgent):
    """Minimal RL agent backed by :class:`TransformerActorCritic`."""

    name = "transformer_rl"

    def __init__(
        self,
        memory_agent: MemoryAgent | None = None,
        *,
        data_url: str = "sqlite:///offline_data.db",
        hidden_size: int = 32,
        nhead: int = 2,
        num_layers: int = 1,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        model_path: str | Path = "transformer_rl.pt",
        replay_url: str = "sqlite:///replay.db",
        device: str = "cpu",
    ) -> None:
        self.memory_agent = memory_agent or MemoryAgent()
        self.offline_data = OfflineData(data_url)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.model_path = Path(model_path)
        self.replay = ReplayBuffer(replay_url)
        self.device = torch.device(device)
        self._last_mtime = 0.0
        self._seen_ids: set[int] = set()
        self._last_id: int = 0
        self._task: asyncio.Task | None = None
        self._logger = logging.getLogger(__name__)

        self.model = TransformerActorCritic(4, hidden_size, nhead, num_layers)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self._fitted = False
        self._jit = None
        if self.model_path.exists():
            self._load_weights()

    def _load_weights(self) -> None:
        script_path = self.model_path.with_suffix(".ptc")
        if script_path.exists():
            try:
                self._jit = torch.jit.load(script_path, map_location=self.device)
                self._last_mtime = os.path.getmtime(script_path)
                self._fitted = True
                return
            except Exception:
                self._jit = None
        data = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(data.get("model_state", {}))
        opt_state = data.get("optim_state")
        if opt_state:
            self.optimizer.load_state_dict(opt_state)
        self._last_mtime = os.path.getmtime(self.model_path)
        self._fitted = True
        self._jit = None

    def reload_weights(self) -> None:
        self._load_weights()

    def _maybe_reload(self) -> None:
        if not self.model_path.exists():
            return
        script = self.model_path.with_suffix(".ptc")
        mtime = os.path.getmtime(script if script.exists() else self.model_path)
        if mtime > self._last_mtime:
            self._load_weights()

    def _state(self, token: str, portfolio: Portfolio) -> List[float]:
        pos = portfolio.balances.get(token)
        amt = float(pos.amount) if pos else 0.0
        depth, imb, _ = snapshot(token)
        regime = detect_regime(portfolio.price_history.get(token, []))
        r = {"bull": 1.0, "bear": -1.0}.get(regime, 0.0)
        return [amt, depth, imb, r]

    def _log_trades(self) -> None:
        trades = self.memory_agent.memory.list_trades(since_id=self._last_id)
        prices: dict[str, list[float]] = {}
        for t in trades:
            tid = getattr(t, "id", None)
            if tid is not None and tid in self._seen_ids:
                continue
            if tid is not None:
                self._seen_ids.add(tid)
                if tid > self._last_id:
                    self._last_id = tid
            reward = float(t.amount) * float(t.price)
            if t.direction == "buy":
                reward = -reward
            seq = prices.setdefault(t.token, [])
            regime_label = detect_regime(seq)
            r = {"bull": 1.0, "bear": -1.0}.get(regime_label, 0.0)
            state = [float(t.amount), float(t.price), 0.0, r]
            self.replay.add(state, t.direction, reward, getattr(t, "emotion", ""), regime_label)
            seq.append(float(t.price))

    def train(self, regime: str | None = None) -> None:
        self._log_trades()
        batch = self.replay.sample(64, regime=regime)
        if not batch:
            return
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([0 if b[1] == "buy" else 1 for b in batch], device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        actor_net = self._jit if self._jit is not None else self.model
        logits, values = actor_net(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        with torch.no_grad():
            advantages = rewards - values
            returns = rewards
        ratio = torch.exp(log_probs - log_probs.detach())
        s1 = ratio * advantages
        s2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(s1, s2).mean()
        critic_loss = self.model.loss_fn(values, returns)
        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._fitted = True
        torch.save({"model_state": self.model.cpu().state_dict(), "optim_state": self.optimizer.state_dict()}, self.model_path)
        self.model.to(self.device)
        self._last_mtime = os.path.getmtime(self.model_path)

    async def _online_loop(self, interval: float = 60.0) -> None:
        while True:
            try:
                self.train()
            except Exception as exc:
                self._logger.error("online train failed: %s", exc)
            await asyncio.sleep(interval)

    def start_online_learning(self, interval: float = 60.0) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._online_loop(interval))

    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        self.start_online_learning()
        self._maybe_reload()
        regime = detect_regime(portfolio.price_history.get(token, []))
        self.train(regime)
        state = torch.tensor([self._state(token, portfolio)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actor_net = self._jit if self._jit is not None else self.model
            logits, _ = actor_net(state)
            logits = logits[0]
        action = "buy" if logits[0] >= logits[1] else "sell"
        if action == "buy":
            return [{"token": token, "side": "buy", "amount": 1.0, "price": 0.0, "agent": self.name}]
        position = portfolio.balances.get(token)
        if position:
            return [{"token": token, "side": "sell", "amount": position.amount, "price": 0.0, "agent": self.name}]
        return []
