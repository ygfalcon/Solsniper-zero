from __future__ import annotations

import asyncio
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - optional dependency
    class _Trainer:
        def __init__(self, *a, **k):
            pass

        def fit(self, model, data):
            pass

    class _LightningModule(nn.Module):
        def save_hyperparameters(self, *args, **kwargs):
            pass

    class _PL:
        LightningModule = _LightningModule
        LightningDataModule = object
        Trainer = _Trainer

    pl = _PL()

from .offline_data import OfflineData
from .simulation import run_simulations, predict_price_movement
from .news import fetch_sentiment
from .regime import detect_regime
from .event_bus import publish


class _TradeDataset(Dataset):
    """Combine offline trades with simulated experiences."""

    def __init__(
        self,
        trades: Iterable[Any],
        snaps: Iterable[Any],
        sims_per_token: int = 10,
        price_model_path: str | None = None,
        regime_weight: float = 1.0,
    ) -> None:
        self.samples: List[Tuple[List[float], int, float]] = []
        self.price_model_path = price_model_path
        self.regime_weight = float(regime_weight)

        snap_map: dict[str, List[Any]] = {}
        for s in snaps:
            snap_map.setdefault(s.token, []).append(s)
        for seq in snap_map.values():
            seq.sort(key=lambda x: x.timestamp)

        feeds = [u for u in os.getenv("NEWS_FEEDS", "").split(",") if u]
        twitter = [u for u in os.getenv("TWITTER_FEEDS", "").split(",") if u]
        discord = [u for u in os.getenv("DISCORD_FEEDS", "").split(",") if u]
        sentiment = 0.0
        if feeds or twitter or discord:
            try:
                sentiment = fetch_sentiment(
                    feeds,
                    twitter_urls=twitter,
                    discord_urls=discord,
                )
            except Exception:
                sentiment = 0.0

        price_hist: dict[str, List[float]] = {}
        for t in trades:
            try:
                pred = predict_price_movement(t.token, model_path=price_model_path)
            except Exception:
                pred = 0.0
            depth = slippage = imbalance = tx_rate = 0.0
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
                depth = float(getattr(snap, "depth", 0.0))
                slippage = float(getattr(snap, "slippage", 0.0))
                imbalance = float(getattr(snap, "imbalance", 0.0))
                tx_rate = float(getattr(snap, "tx_rate", 0.0))
            hist = price_hist.setdefault(t.token, [])
            regime = detect_regime(hist)
            r = {"bull": 1.0, "bear": -1.0}.get(regime, 0.0) * self.regime_weight
            state = [
                float(t.price),
                float(getattr(t, "amount", 0.0)),
                depth,
                slippage,
                tx_rate,
                sentiment,
                imbalance,
                pred,
                r,
            ]
            reward = float(t.amount) * float(t.price)
            side = getattr(t, "side", getattr(t, "direction", "buy"))
            action = 0 if side == "buy" else 1
            if side == "buy":
                reward = -reward
            self.samples.append((state, action, reward))
            hist.append(float(t.price))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, action, reward = self.samples[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
        )


class TradeDataModule(pl.LightningDataModule):
    """PyTorch Lightning datamodule wrapping :class:`_TradeDataset`."""

    def __init__(self, db_url: str, batch_size: int = 64, sims_per_token: int = 10, *, price_model_path: str | None = None, regime_weight: float = 1.0) -> None:
        super().__init__()
        self.db_url = db_url
        self.batch_size = batch_size
        self.sims_per_token = sims_per_token
        self.price_model_path = price_model_path
        self.regime_weight = float(regime_weight)
        self.dataset: _TradeDataset | None = None

    def setup(self, stage: str | None = None) -> None:  # pragma: no cover - simple
        data = OfflineData(self.db_url)
        trades = data.list_trades()
        snaps = data.list_snapshots()
        self.dataset = _TradeDataset(
            trades,
            snaps,
            self.sims_per_token,
            price_model_path=self.price_model_path,
            regime_weight=self.regime_weight,
        )

    def train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            self.setup()
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() or 1,
        )


class LightningPPO(pl.LightningModule):
    """Minimal PPO actor-critic implementation."""

    def __init__(self, hidden_size: int = 32, lr: float = 3e-4, gamma: float = 0.99, clip_epsilon: float = 0.2) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.actor = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.actor(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        states, actions, rewards = batch
        dist = torch.distributions.Categorical(logits=self.actor(states))
        log_probs = dist.log_prob(actions)
        values = self.critic(states).squeeze()
        with torch.no_grad():
            advantages = rewards - values
            returns = rewards
        ratio = torch.exp(log_probs - log_probs.detach())
        s1 = ratio * advantages
        s2 = torch.clamp(ratio, 1 - self.hparams.clip_epsilon, 1 + self.hparams.clip_epsilon) * advantages
        actor_loss = -torch.min(s1, s2).mean()
        critic_loss = self.loss_fn(values, returns)
        loss = actor_loss + 0.5 * critic_loss
        self.log("loss", loss)
        self.log("reward", rewards.mean())
        return loss

    def configure_optimizers(self):  # pragma: no cover - simple
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class LightningDQN(pl.LightningModule):
    """Simple DQN model trained via mean squared error."""

    def __init__(self, hidden_size: int = 32, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        states, actions, rewards = batch
        q = self.model(states)
        target = q.detach().clone()
        for i, a, r in zip(range(len(actions)), actions, rewards):
            target[i, a] = r
        loss = self.loss_fn(q, target)
        self.log("loss", loss)
        self.log("reward", rewards.mean())
        return loss

    def configure_optimizers(self):  # pragma: no cover - simple
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class _MetricsCallback(pl.callbacks.Callback):
    """Callback publishing metrics after each training epoch."""

    def on_train_epoch_end(self, trainer, pl_module):  # pragma: no cover - simple
        loss = trainer.callback_metrics.get("loss")
        reward = trainer.callback_metrics.get("reward")
        if loss is None or reward is None:
            return
        try:
            loss_val = float(loss)
        except Exception:
            loss_val = float(loss.item())
        try:
            reward_val = float(reward)
        except Exception:
            reward_val = float(reward.item())
        publish("rl_metrics", {"loss": loss_val, "reward": reward_val})


class RLTraining:
    """Utility class that trains RL agents from ``offline_data.db``."""

    def __init__(
        self,
        *,
        db_url: str = "sqlite:///offline_data.db",
        model_path: str | Path = "ppo_model.pt",
        algo: str = "ppo",
        batch_size: int = 64,
        sims_per_token: int = 10,
        price_model_path: str | None = None,
        regime_weight: float = 1.0,
        device: str | None = None,
        metrics_url: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.data = TradeDataModule(
            db_url,
            batch_size=batch_size,
            sims_per_token=sims_per_token,
            price_model_path=price_model_path,
            regime_weight=regime_weight,
        )
        if algo == "dqn":
            self.model: pl.LightningModule = LightningDQN()
        else:
            self.model = LightningPPO()
        if self.model_path.exists():
            try:
                self.model.load_state_dict(torch.load(self.model_path))
            except Exception:  # pragma: no cover - corrupt file
                pass
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        acc = "gpu" if device == "cuda" else device
        kwargs = dict(max_epochs=3, accelerator=acc, enable_progress_bar=False)
        if acc != "cpu":
            kwargs["devices"] = 1
        self.trainer = pl.Trainer(callbacks=[_MetricsCallback()], **kwargs)
        self._task: asyncio.Task | None = None
        self._logger = logging.getLogger(__name__)
        from .metrics_client import start_metrics_exporter
        self._metrics_sub = start_metrics_exporter(metrics_url)

    def close(self) -> None:
        try:
            self._metrics_sub()
        except Exception:
            pass

    def train(self) -> None:
        """Run one training cycle and persist weights."""
        self.data.setup()
        self.trainer.fit(self.model, self.data)
        torch.save(self.model.state_dict(), self.model_path)

    async def _loop(self, interval: float) -> None:
        while True:
            try:
                self.train()
            except Exception as exc:  # pragma: no cover - logging
                self._logger.error("periodic training failed: %s", exc)
            await asyncio.sleep(interval)

    def start_periodic_retraining(self, interval: float = 3600.0) -> asyncio.Task:
        """Begin background task that periodically retrains the model."""
        if self._task is None:
            self._task = asyncio.create_task(self._loop(interval))
        return self._task


def fit(
    trades: Iterable[Any],
    snaps: Iterable[Any],
    *,
    model_path: str | Path = "ppo_model.pt",
    algo: str = "ppo",
    regime_weight: float = 1.0,
    device: str | None = None,
) -> None:
    """Train a lightweight RL model from in-memory samples.

    Parameters
    ----------
    trades:
        Iterable of trade records as returned by :class:`~solhunter_zero.memory.Memory`.
    snaps:
        Iterable of :class:`~solhunter_zero.offline_data.MarketSnapshot`.
    model_path:
        File path where the checkpoint is stored.
    algo:
        ``"ppo"`` or ``"dqn"`` model type.
    device:
        Optional accelerator string, ``"cuda"`` or ``"mps"``.
    """

    dataset = _TradeDataset(trades, snaps, regime_weight=regime_weight)
    if len(dataset) == 0:
        Path(model_path).touch()
        return

    if algo == "dqn":
        model: pl.LightningModule = LightningDQN()
    else:
        model = LightningPPO()

    path = Path(model_path)
    if path.exists():
        try:
            model.load_state_dict(torch.load(path))
        except Exception:  # pragma: no cover - ignore corrupt weights
            pass

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    acc = "gpu" if device == "cuda" else device
    kwargs = dict(max_epochs=3, accelerator=acc, enable_progress_bar=False)
    if acc != "cpu":
        kwargs["devices"] = 1
    trainer = pl.Trainer(callbacks=[_MetricsCallback()], **kwargs)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=os.cpu_count() or 1,
    )
    trainer.fit(model, train_dataloaders=loader)
    torch.save(model.state_dict(), path)
