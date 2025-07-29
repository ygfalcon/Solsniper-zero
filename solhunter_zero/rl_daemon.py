from __future__ import annotations

import asyncio
import logging
import threading
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Any
import os
import time
import types

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .base_memory import BaseMemory
from .memory import Memory, Trade
from .offline_data import OfflineData, MarketSnapshot
from . import rl_training
from .risk import average_correlation
from .event_bus import subscription, publish
from .schemas import ActionExecuted, RLCheckpoint, RLWeights

logger = logging.getLogger(__name__)


class _TradeDataset(Dataset):
    """Dataset wrapping trade history with risk and market metrics."""

    def __init__(self, trades: Iterable, snaps: Iterable, metrics: Tuple[float, float, float]) -> None:
        self.samples: List[Tuple[List[float], int, float]] = []
        snap_map: dict[str, List[Any]] = {}
        for s in snaps:
            snap_map.setdefault(s.token, []).append(s)
        for seq in snap_map.values():
            seq.sort(key=lambda x: x.timestamp)

        drawdown, volatility, corr = metrics
        for t in trades:
            mempool = trend = depth = 0.0
            seq = snap_map.get(t.token)
            if seq:
                idx = 0
                ts = getattr(t, "timestamp", None)
                for i, s in enumerate(seq):
                    if ts is None or s.timestamp <= ts:
                        idx = i
                    else:
                        break
                snap = seq[idx]
                mempool = float(getattr(snap, "tx_rate", 0.0))
                depth = float(getattr(snap, "depth", 0.0))
                if idx > 0:
                    prev = seq[idx - 1]
                    p0 = float(prev.price)
                    if p0:
                        trend = (float(snap.price) - p0) / p0

            state = [
                float(t.price),
                float(getattr(t, "amount", 0.0)),
                drawdown,
                volatility,
                corr,
                mempool,
                trend,
                depth,
            ]
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
    def __init__(self, input_size: int = 8, hidden_size: int = 32) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.model(x)


class _PPO(nn.Module):
    def __init__(self, input_size: int = 8, hidden_size: int = 32, clip_epsilon: float = 0.2) -> None:
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
    workers = int(os.getenv("DL_WORKERS", os.cpu_count() or 1))
    loader = DataLoader(
        data,
        batch_size=32,
        shuffle=True,
        num_workers=workers,
    )
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
    workers = int(os.getenv("DL_WORKERS", os.cpu_count() or 1))
    loader = DataLoader(
        data,
        batch_size=32,
        shuffle=True,
        num_workers=workers,
    )
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
        memory: BaseMemory | None = None,
        memory_path: str = "sqlite:///memory.db",
        data_path: str = "offline_data.db",
        model_path: str | Path = "ppo_model.pt",
        algo: str = "ppo",
        device: str | None = None,
        *,
        agents: Iterable[Any] | None = None,
        queue: asyncio.Queue | None = None,
        metrics_url: str | None = None,
    ) -> None:
        self.memory = memory or Memory(memory_path)
        self.data_path = data_path
        self.data = OfflineData(f"sqlite:///{data_path}")
        self.model_path = Path(model_path)
        self.algo = algo
        self.last_train_time: float | None = None
        self.checkpoint_path: str = str(self.model_path)
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
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
        self._proc: subprocess.Popen | None = None
        self._last_trade_id = 0
        self._last_snap_id = 0
        self.queue = queue
        self.current_risk = float(os.getenv("RISK_MULTIPLIER", "1.0"))
        self._subscriptions: list[Any] = []
        if queue is not None:
            async def _enqueue(payload):
                if isinstance(payload, ActionExecuted):
                    await queue.put(payload.action)
                else:
                    await queue.put(payload)
                await asyncio.to_thread(self.train)

            sub = subscription("action_executed", _enqueue)
            sub.__enter__()
            self._subscriptions.append(sub)

        async def _update_risk(payload):
            self.current_risk = float(payload.get("multiplier", self.current_risk))
            logger.info("risk multiplier updated to %s", self.current_risk)

        risk_sub = subscription("risk_updated", _update_risk)
        risk_sub.__enter__()
        self._subscriptions.append(risk_sub)
        from .metrics_client import start_metrics_exporter
        sub = start_metrics_exporter(metrics_url)
        self._subscriptions.append(types.SimpleNamespace(__exit__=lambda *a, **k: sub()))

    def close(self) -> None:
        for sub in self._subscriptions:
            sub.__exit__(None, None, None)

    def _fetch_new(self) -> tuple[list[Trade], list[MarketSnapshot]]:
        """Return new trades and snapshots since the last training cycle."""
        trades: list[Trade] = []
        snaps: list[MarketSnapshot] = []
        with self.memory.Session() as session:
            q = session.query(Trade).filter(Trade.id > self._last_trade_id)
            trades = list(q.order_by(Trade.id))
            if trades:
                self._last_trade_id = trades[-1].id
        with self.data.Session() as session:
            q = session.query(MarketSnapshot).filter(MarketSnapshot.id > self._last_snap_id)
            snaps = list(q.order_by(MarketSnapshot.id))
            if snaps:
                self._last_snap_id = snaps[-1].id
        if self.queue is not None:
            from types import SimpleNamespace
            while True:
                try:
                    item = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if isinstance(item, dict):
                    trade = SimpleNamespace(
                        token=item.get("token"),
                        direction=item.get("side"),
                        amount=float(item.get("amount", 0.0)),
                        price=float(item.get("price", 0.0)),
                    )
                else:
                    trade = item
                trades.append(trade)
        return trades, snaps

    def register_agent(self, agent: Any) -> None:
        """Register an agent to reload checkpoints after training."""
        self.agents.append(agent)

    def train(self) -> None:
        trades, snaps = self._fetch_new()
        if not trades and not snaps:
            return
        rl_training.fit(
            trades,
            snaps,
            model_path=self.model_path,
            algo=self.algo,
            device=self.device.type,
        )
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        except Exception as exc:  # pragma: no cover - corrupt file
            logger.error("failed to load updated model: %s", exc)
        for ag in self.agents:
            try:
                if hasattr(ag, "reload_weights"):
                    ag.reload_weights()
                else:
                    ag._load_weights()
            except Exception:  # pragma: no cover - ignore bad agents
                logger.error("failed to reload agent")
        self.last_train_time = time.time()
        self.checkpoint_path = str(self.model_path)
        logger.info("saved checkpoint to %s", self.model_path)
        publish(
            "rl_checkpoint",
            RLCheckpoint(time=self.last_train_time, path=self.checkpoint_path),
        )
        publish(
            "rl_weights",
            RLWeights(
                weights={},
                risk={"risk_multiplier": self.current_risk},
            ),
        )
        reward = 0.0
        for t in trades:
            val = float(getattr(t, "amount", 0.0)) * float(getattr(t, "price", 0.0))
            if getattr(t, "direction", "buy") == "sell":
                reward += val
            else:
                reward -= val
        publish("rl_metrics", {"loss": 0.0, "reward": reward})

    async def _loop(self, interval: float) -> None:
        while True:
            try:
                self.train()
            except Exception as exc:  # pragma: no cover - log errors
                logger.error("daemon training failed: %s", exc)
            await asyncio.sleep(interval)

    async def _watch_external(self, interval: float) -> None:
        last = self.model_path.stat().st_mtime if self.model_path.exists() else 0.0
        while True:
            await asyncio.sleep(interval)
            try:
                mtime = self.model_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if mtime > last:
                last = mtime
                try:
                    self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                except Exception as exc:  # pragma: no cover - corrupt file
                    logger.error("failed to load updated model: %s", exc)
                    continue
                for ag in self.agents:
                    try:
                        if hasattr(ag, "reload_weights"):
                            ag.reload_weights()
                        else:
                            ag._load_weights()
                    except Exception:
                        logger.error("failed to reload agent")
                self.last_train_time = time.time()
                self.checkpoint_path = str(self.model_path)
                logger.info("reloaded checkpoint from %s", self.model_path)

    def start(
        self,
        interval: float = 3600.0,
        *,
        auto_train: bool = False,
        tune_interval: float | None = None,
    ) -> asyncio.Task:
        """Begin the training loop in the current or a background event loop."""
        if self._task is not None:
            return self._task

        # Auto-select accelerator if none was provided
        if self.device.type == "cpu":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            if self.device.type != "cpu":
                self.model.to(self.device)

        if auto_train:
            if tune_interval is None:
                tune_interval = interval
            script = Path(__file__).resolve().parent.parent / "scripts" / "auto_train_rl.py"
            self._proc = subprocess.Popen(
                [sys.executable, str(script), "--db", self.data_path, "--model", str(self.model_path), "--interval", str(tune_interval)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # pragma: no cover - no running loop
                loop = asyncio.new_event_loop()
                t = threading.Thread(target=loop.run_forever, daemon=True)
                t.start()
            self._task = loop.create_task(self._watch_external(interval))
            return self._task

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - no running loop
            loop = asyncio.new_event_loop()
            t = threading.Thread(target=loop.run_forever, daemon=True)
            t.start()
        self._task = loop.create_task(self._loop(interval))
        return self._task

