from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .memory import Memory
from .offline_data import OfflineData
from .risk import average_correlation

logger = logging.getLogger(__name__)


class _TradeDataset(Dataset):
    """Dataset wrapping trade history with risk metrics."""

    def __init__(self, trades: Iterable, metrics: Tuple[float, float, float]) -> None:
        self.samples: List[Tuple[List[float], int, float]] = []
        drawdown, volatility, corr = metrics
        for t in trades:
            state = [float(t.price), float(getattr(t, "amount", 0.0)), drawdown, volatility, corr]
            reward = float(t.amount) * float(t.price)
            action = 0 if t.direction == "buy" else 1
            if t.direction == "buy":
                reward = -reward
            self.samples.append((state, action, reward))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        state, action, reward = self.samples[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
        )


class _DQN(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 32) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.model(x)


class _PPO(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 32, clip_epsilon: float = 0.2) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.clip_epsilon = clip_epsilon
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.actor(x)


def _metrics(trades: Iterable, snaps: Iterable) -> Tuple[float, float, float]:
    price_hist: dict[str, List[float]] = {}
    for s in snaps:
        price_hist.setdefault(s.token, []).append(float(s.price))

    bal = high = 0.0
    for t in trades:
        val = float(t.amount) * float(t.price)
        if t.direction == "buy":
            bal -= val
        else:
            bal += val
        if bal > high:
            high = bal
    drawdown = (high - bal) / high if high > 0 else 0.0

    vols = []
    for seq in price_hist.values():
        if len(seq) >= 2:
            arr = torch.tensor(seq[-30:], dtype=torch.float32)
            vols.append(float(arr.std() / (arr.mean() + 1e-8)))
    volatility = sum(vols) / len(vols) if vols else 0.0

    corr = average_correlation(price_hist) if len(price_hist) > 1 else 0.0
    return drawdown, volatility, corr


def _train_dqn(model: _DQN, data: Dataset, device: torch.device) -> None:
    loader = DataLoader(data, batch_size=32, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(3):
        for states, actions, rewards in loader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            q = model(states)
            target = q.detach().clone()
            for i, a, r in zip(range(len(actions)), actions, rewards):
                target[i, a] = r
            loss = loss_fn(q, target)
            opt.zero_grad()
            loss.backward()
            opt.step()


def _train_ppo(model: _PPO, data: Dataset, device: torch.device) -> None:
    loader = DataLoader(data, batch_size=32, shuffle=True)
    opt = torch.optim.Adam(list(model.actor.parameters()) + list(model.critic.parameters()), lr=3e-4)
    model.train()
    for _ in range(3):
        for states, actions, rewards in loader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            dist = torch.distributions.Categorical(logits=model.actor(states))
            log_probs = dist.log_prob(actions)
            values = model.critic(states).squeeze()
            with torch.no_grad():
                advantages = rewards - values
                returns = rewards
            ratio = torch.exp(log_probs - log_probs.detach())
            s1 = ratio * advantages
            s2 = torch.clamp(ratio, 1 - model.clip_epsilon, 1 + model.clip_epsilon) * advantages
            actor_loss = -torch.min(s1, s2).mean()
            critic_loss = model.loss_fn(values, returns)
            loss = actor_loss + 0.5 * critic_loss
            opt.zero_grad()
            loss.backward()
            opt.step()


class RLDaemon:
    """Background trainer that updates RL models from trade history."""

    def __init__(
        self,
        memory_path: str = "sqlite:///memory.db",
        data_path: str = "offline_data.db",
        model_path: str | Path = "ppo_model.pt",
        algo: str = "ppo",
        device: str | None = None,
        *,
        agents: Iterable[Any] | None = None,
    ) -> None:
        self.memory = Memory(memory_path)
        self.data = OfflineData(f"sqlite:///{data_path}")
        self.model_path = Path(model_path)
        self.algo = algo
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if algo == "dqn":
            self.model: nn.Module = _DQN()
        else:
            self.model = _PPO()
        if self.model_path.exists():
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            except Exception:  # pragma: no cover - corrupt file
                pass
        self.model.to(self.device)
        self.agents: List[Any] = list(agents) if agents else []
        self._task: asyncio.Task | None = None

    def register_agent(self, agent: Any) -> None:
        """Register an agent to reload checkpoints after training."""
        self.agents.append(agent)

    def train(self) -> None:
        trades = self.memory.list_trades()
        snaps = self.data.list_snapshots()
        metrics = _metrics(trades, snaps)
        dataset = _TradeDataset(trades, metrics)
        if not dataset:
            return
        if self.algo == "dqn":
            _train_dqn(self.model, dataset, self.device)
        else:
            _train_ppo(self.model, dataset, self.device)
        torch.save(self.model.state_dict(), self.model_path)
        for ag in self.agents:
            try:
                ag._load_weights()
            except Exception:  # pragma: no cover - ignore bad agents
                logger.error("failed to reload agent")
        logger.info("saved checkpoint to %s", self.model_path)

    async def _loop(self, interval: float) -> None:
        while True:
            try:
                self.train()
            except Exception as exc:  # pragma: no cover - log errors
                logger.error("daemon training failed: %s", exc)
            await asyncio.sleep(interval)

    def start(self, interval: float = 3600.0) -> asyncio.Task:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(interval))
        return self._task
