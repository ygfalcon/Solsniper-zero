from __future__ import annotations

import asyncio
import logging
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


class _TradeDataset(Dataset):
    """Combine offline trades with simulated experiences."""

    def __init__(
        self,
        trades: Iterable[Any],
        tokens: Iterable[str],
        sims_per_token: int = 10,
        price_model_path: str | None = None,
    ) -> None:
        self.samples: List[Tuple[List[float], int, float]] = []
        self.price_model_path = price_model_path
        for t in trades:
            try:
                pred = predict_price_movement(t.token, model_path=price_model_path)
            except Exception:
                pred = 0.0
            state = [float(t.price), float(getattr(t, "amount", 0.0)), 0.0, pred]
            reward = float(t.amount) * float(t.price)
            action = 0 if t.side == "buy" else 1
            if t.side == "buy":
                reward = -reward
            self.samples.append((state, action, reward))
        for tok in set(tokens):
            sims = []
            for s in sims:
                try:
                    pred = predict_price_movement(tok, model_path=price_model_path)
                except Exception:
                    pred = 0.0
                state = [s.liquidity, s.slippage, s.volatility, pred]
                action = 0 if s.expected_roi >= 0 else 1
                self.samples.append((state, action, s.expected_roi))

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

    def __init__(self, db_url: str, batch_size: int = 64, sims_per_token: int = 10, *, price_model_path: str | None = None) -> None:
        super().__init__()
        self.db_url = db_url
        self.batch_size = batch_size
        self.sims_per_token = sims_per_token
        self.price_model_path = price_model_path
        self.dataset: _TradeDataset | None = None

    def setup(self, stage: str | None = None) -> None:  # pragma: no cover - simple
        data = OfflineData(self.db_url)
        trades = data.list_trades()
        tokens = {t.token for t in trades}
        snaps = data.list_snapshots()
        tokens.update(s.token for s in snaps)
        self.dataset = _TradeDataset(trades, tokens, self.sims_per_token, price_model_path=self.price_model_path)

    def train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            self.setup()
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


class LightningPPO(pl.LightningModule):
    """Minimal PPO actor-critic implementation."""

    def __init__(self, hidden_size: int = 32, lr: float = 3e-4, gamma: float = 0.99, clip_epsilon: float = 0.2) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.actor = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(4, hidden_size),
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
        return loss

    def configure_optimizers(self):  # pragma: no cover - simple
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class LightningDQN(pl.LightningModule):
    """Simple DQN model trained via mean squared error."""

    def __init__(self, hidden_size: int = 32, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(4, hidden_size),
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
        return loss

    def configure_optimizers(self):  # pragma: no cover - simple
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


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
    ) -> None:
        self.model_path = Path(model_path)
        self.data = TradeDataModule(
            db_url,
            batch_size=batch_size,
            sims_per_token=sims_per_token,
            price_model_path=price_model_path,
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
        self.trainer = pl.Trainer(max_epochs=3, accelerator="auto", enable_progress_bar=False)
        self._task: asyncio.Task | None = None
        self._logger = logging.getLogger(__name__)

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

    def start_periodic_retraining(self, interval: float = 3600.0) -> None:
        """Begin background task that periodically retrains the model."""
        if self._task is None:
            self._task = asyncio.create_task(self._loop(interval))
