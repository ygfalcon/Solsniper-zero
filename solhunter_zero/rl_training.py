from __future__ import annotations

import asyncio
import logging
import os
import datetime
from pathlib import Path
from typing import Any, Iterable, List, Tuple
import time
import psutil
from collections import deque
from .multi_rl import PopulationRL
from .advanced_memory import AdvancedMemory

try:
    from numba import njit  # type: ignore
    _NUMBA = True
except Exception:  # pragma: no cover - optional
    _NUMBA = False

    def njit(*a, **k):  # type: ignore
        def wrapper(fn):
            return fn

        return wrapper

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
try:
    from torch.utils.data import WeightedRandomSampler
except Exception:  # pragma: no cover - optional dependency
    WeightedRandomSampler = None  # type: ignore
from typing import Callable
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
from .simulation import predict_price_movement
from .news import fetch_sentiment
from .event_bus import publish, subscription, connect_broker
from .event_bus import _BROKER_URL  # type: ignore
from .config import get_broker_url


_CPU_USAGE = 0.0
_CPU_SUB = None


def _get_cpu_usage(callback: Callable[[], float] | None = None) -> float:
    """Return latest CPU usage from callback or ``system_metrics`` events."""
    global _CPU_SUB, _CPU_USAGE
    if callback is not None:
        try:
            return float(callback())
        except Exception:
            return 0.0
    if _CPU_SUB is None:
        def _update(payload: Any) -> None:
            global _CPU_USAGE
            try:
                _CPU_USAGE = float(payload.get("cpu", payload.get("usage", payload)))
            except Exception:
                pass

        sub = subscription("system_metrics", _update)
        sub.__enter__()
        _CPU_SUB = sub
    return _CPU_USAGE


def _calc_num_workers(
    size: int,
    *,
    dynamic: bool = False,
    cpu_callback: Callable[[], float] | None = None,
) -> int:
    env_val = os.getenv("RL_NUM_WORKERS")
    if env_val is not None:
        try:
            return int(env_val)
        except Exception:
            pass
    base = min(os.cpu_count() or 1, max(1, size // 100))
    if dynamic:
        usage = _get_cpu_usage(cpu_callback)
        try:
            frac = 1.0 - float(usage) / 100.0
        except Exception:
            frac = 1.0
        base = max(1, int(base * max(0.1, frac)))
    try:
        mem = float(psutil.virtual_memory().percent)
    except Exception:
        mem = 0.0
    if mem > 80.0:
        base = max(1, base // 2)
    return base


def _ensure_mmap_dataset(db_url: str, out_path: Path) -> None:
    """Create ``out_path`` using ``build_mmap_dataset`` if it doesn't exist."""
    if out_path.exists():
        return
    try:
        from scripts import build_mmap_dataset
    except Exception:  # pragma: no cover - missing script
        return
    db_path = db_url
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
    try:
        build_mmap_dataset.main(["--db", db_path, "--out", str(out_path)])
    except Exception as exc:  # pragma: no cover - generation failure
        logging.getLogger(__name__).warning(
            "failed to build mmap dataset: %s", exc
        )


@njit(cache=True)
def _compute_regimes_nb(prices: np.ndarray, inv: np.ndarray, first_price: np.ndarray, weight: float, thr: float) -> np.ndarray:
    n = prices.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        base = first_price[inv[i]]
        change = prices[i] / base - 1.0
        if change > thr:
            out[i] = weight
        elif change < -thr:
            out[i] = -weight
    return out


def _compute_regimes(tokens: np.ndarray, prices: np.ndarray, weight: float, thr: float = 0.02) -> np.ndarray:
    uniq, first_idx, inv = np.unique(tokens, return_index=True, return_inverse=True)
    first_price = prices[first_idx]
    if _NUMBA:
        return _compute_regimes_nb(prices.astype(np.float32), inv.astype(np.int64), first_price.astype(np.float32), weight, thr)
    base = first_price[inv]
    change = prices / base - 1.0
    res = np.where(change > thr, 1.0, np.where(change < -thr, -1.0, 0.0))
    return res.astype(np.float32) * weight


class LiveTradeDataset:
    """Buffer trade and depth events from :mod:`event_bus`.

    When ``broker_url`` is provided the dataset connects to the message
    broker so events from remote peers are received as well.
    """

    def __init__(self, capacity: int = 10000, *, broker_url: str | None = None) -> None:
        self.capacity = int(capacity)
        self.trades: deque[Any] = deque(maxlen=self.capacity)
        self.snaps: deque[Any] = deque(maxlen=self.capacity)
        self._last_trade = 0
        self._last_snap = 0

        def _add_trade(payload: Any) -> None:
            from types import SimpleNamespace

            self.trades.append(
                SimpleNamespace(
                    token=getattr(payload, "token", None) or payload.get("token"),
                    direction=getattr(payload, "direction", None)
                    or payload.get("direction"),
                    amount=float(getattr(payload, "amount", payload.get("amount", 0.0))),
                    price=float(getattr(payload, "price", payload.get("price", 0.0))),
                    timestamp=getattr(payload, "timestamp", datetime.datetime.utcnow()),
                )
            )

        def _add_depth(payload: Any) -> None:
            from types import SimpleNamespace

            entries = getattr(payload, "entries", None) or payload.get("entries", {})
            ts = getattr(payload, "ts", None)
            for tok, info in entries.items():
                bids = float(getattr(info, "bids", info.get("bids", 0.0)))
                asks = float(getattr(info, "asks", info.get("asks", 0.0)))
                tx = float(getattr(info, "tx_rate", info.get("tx_rate", 0.0)))
                t = datetime.datetime.utcfromtimestamp(float(ts or getattr(info, "ts", time.time())))
                self.snaps.append(
                    SimpleNamespace(
                        token=tok,
                        price=float(getattr(info, "price", info.get("price", 0.0))),
                        depth=bids + asks,
                        total_depth=bids + asks,
                        slippage=0.0,
                        volume=0.0,
                        imbalance=(bids - asks),
                        tx_rate=tx,
                        whale_share=0.0,
                        spread=0.0,
                        sentiment=0.0,
                        timestamp=t,
                    )
                )

        self._trade_sub = subscription("trade_logged", _add_trade)
        self._trade_sub.__enter__()
        self._depth_sub = subscription("depth_update", _add_depth)
        self._depth_sub.__enter__()

        url = broker_url or get_broker_url()
        if url and _BROKER_URL is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop:
                loop.create_task(connect_broker(url))
            else:
                asyncio.run(connect_broker(url))

    def close(self) -> None:
        self._trade_sub.__exit__(None, None, None)
        self._depth_sub.__exit__(None, None, None)

    def fetch_new(self) -> tuple[list[Any], list[Any]]:
        trades = list(self.trades)[self._last_trade :]
        snaps = list(self.snaps)[self._last_snap :]
        self._last_trade = len(self.trades)
        self._last_snap = len(self.snaps)
        return trades, snaps


class _TradeDataset(Dataset):
    """Combine offline trades with simulated experiences."""

    def __init__(
        self,
        trades: Iterable[Any],
        snaps: Iterable[Any],
        sims_per_token: int = 10,
        price_model_path: str | None = None,
        regime_weight: float = 1.0,
        memory: AdvancedMemory | None = None,
    ) -> None:
        start_time = time.time()
        self.price_model_path = price_model_path
        self.regime_weight = float(regime_weight)

        trade_list = list(trades)
        n = len(trade_list)

        clusters = np.zeros(n, dtype=np.float32)
        if memory is not None:
            try:
                memory.cluster_trades()
            except Exception:
                pass
            try:
                centroids = getattr(memory, "cluster_centroids", None)
                num_clusters = len(centroids) if centroids is not None else 0
            except Exception:
                num_clusters = 0
            for i, t in enumerate(trade_list):
                try:
                    cid = memory.top_cluster(getattr(t, "context", ""))
                except Exception:
                    cid = None
                if cid is not None and num_clusters:
                    clusters[i] = float(cid) / float(num_clusters)

        tokens = np.array([t.token for t in trade_list])
        prices = np.array([float(t.price) for t in trade_list], dtype=np.float32)
        amounts = np.array(
            [float(getattr(t, "amount", 0.0)) for t in trade_list], dtype=np.float32
        )
        timestamps = np.array(
            [getattr(t, "timestamp", datetime.datetime.utcnow()).timestamp() for t in trade_list],
            dtype=np.float64,
        )
        sides = [getattr(t, "side", getattr(t, "direction", "buy")) for t in trade_list]
        actions = np.array([0 if s == "buy" else 1 for s in sides], dtype=np.int64)

        snap_tokens = np.array([s.token for s in snaps], dtype=object)
        snap_ts = np.array([s.timestamp.timestamp() for s in snaps], dtype=np.float64)
        snap_depth = np.array(
            [float(getattr(s, "depth", 0.0)) for s in snaps], dtype=np.float32
        )
        snap_slip = np.array(
            [float(getattr(s, "slippage", 0.0)) for s in snaps], dtype=np.float32
        )
        snap_imb = np.array(
            [float(getattr(s, "imbalance", 0.0)) for s in snaps], dtype=np.float32
        )
        snap_tx = np.array(
            [float(getattr(s, "tx_rate", 0.0)) for s in snaps], dtype=np.float32
        )

        if snap_ts.size:
            order = np.lexsort((snap_ts, snap_tokens))
            snap_tokens = snap_tokens[order]
            snap_ts = snap_ts[order]
            snap_depth = snap_depth[order]
            snap_slip = snap_slip[order]
            snap_imb = snap_imb[order]
            snap_tx = snap_tx[order]

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

        depth = np.zeros(n, dtype=np.float32)
        slippage = np.zeros(n, dtype=np.float32)
        imbalance = np.zeros(n, dtype=np.float32)
        tx_rate = np.zeros(n, dtype=np.float32)

        # Vectorized lookup of snapshot features for each trade
        if snap_ts.size:
            uniq_snap_tokens, snap_inverse = np.unique(snap_tokens, return_inverse=True)
            trade_tok_idx = np.searchsorted(uniq_snap_tokens, tokens)
            valid = (trade_tok_idx < len(uniq_snap_tokens)) & (
                uniq_snap_tokens[trade_tok_idx] == tokens
            )

            shift = snap_ts.max() + 1.0
            encoded_snaps = snap_inverse * shift + snap_ts
            order = np.argsort(encoded_snaps)
            encoded_snaps = encoded_snaps[order]
            snap_depth = snap_depth[order]
            snap_slip = snap_slip[order]
            snap_imb = snap_imb[order]
            snap_tx = snap_tx[order]

            encoded_trades = trade_tok_idx * shift + timestamps
            idx = np.searchsorted(encoded_snaps, encoded_trades, side="right") - 1
            idx[idx < 0] = 0
            depth[valid] = snap_depth[idx[valid]]
            slippage[valid] = snap_slip[idx[valid]]
            imbalance[valid] = snap_imb[idx[valid]]
            tx_rate[valid] = snap_tx[idx[valid]]

        uniq_tokens, inverse = np.unique(tokens, return_inverse=True)
        pred_vals = np.zeros(len(uniq_tokens), dtype=np.float32)
        for i, tok in enumerate(uniq_tokens):
            try:
                pred_vals[i] = predict_price_movement(tok, model_path=price_model_path)
            except Exception:
                pred_vals[i] = 0.0
        preds = pred_vals[inverse]

        regimes = _compute_regimes(tokens, prices, self.regime_weight)

        rewards = prices * amounts
        rewards[actions == 0] *= -1
        states = np.column_stack(
            [
                prices,
                amounts,
                depth,
                slippage,
                tx_rate,
                np.full(n, float(sentiment), dtype=np.float32),
                imbalance,
                preds,
                regimes,
            ]
        )
        if memory is not None:
            states = np.column_stack([states, clusters])

        self.states = states
        self.actions = actions
        self.rewards = rewards
        pri = np.abs(rewards).astype(np.float32)
        if pri.sum() == 0:
            pri[:] = 1.0
        self.priorities = pri
        self.build_time = time.time() - start_time
        logging.getLogger(__name__).debug(
            "constructed trade dataset in %.3fs", self.build_time
        )

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.states[idx], dtype=torch.float32),
            torch.tensor(self.actions[idx], dtype=torch.long),
            torch.tensor(self.rewards[idx], dtype=torch.float32),
        )


class PrioritizedReplayDataset(Dataset):
    """Wrap a dataset and expose sampling weights."""

    def __init__(self, dataset: Dataset, priorities: Iterable[float] | None = None) -> None:
        self.dataset = dataset
        if priorities is None:
            priorities = getattr(dataset, "priorities", None)
        if priorities is None:
            priorities = [1.0] * len(dataset)
        weights = torch.tensor(list(priorities), dtype=torch.float32)
        if weights.sum() <= 0:
            weights[:] = 1.0
        self.weights = weights
        if WeightedRandomSampler is not None:
            self.sampler = WeightedRandomSampler(self.weights, len(self.weights), replacement=True)
        else:  # pragma: no cover - optional dependency
            self.sampler = None  # type: ignore

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


class TradeDataModule(pl.LightningDataModule):
    """PyTorch Lightning datamodule wrapping :class:`_TradeDataset`."""

    def __init__(
        self,
        db_url: str,
        batch_size: int = 64,
        sims_per_token: int = 10,
        *,
        price_model_path: str | None = None,
        regime_weight: float = 1.0,
        mmap_path: str | None = None,
        num_workers: int | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        dynamic_workers: bool = False,
        prioritized_replay: bool | None = None,
        cpu_callback: Callable[[], float] | None = None,
    ) -> None:
        super().__init__()
        self.db_url = db_url
        if mmap_path is None:
            default = Path("datasets/offline_data.npz")
            if default.exists():
                mmap_path = str(default)
        self.mmap_path = mmap_path
        self.batch_size = batch_size
        self.sims_per_token = sims_per_token
        self.price_model_path = price_model_path
        self.regime_weight = float(regime_weight)
        env_val = os.getenv("RL_NUM_WORKERS")
        if env_val is not None:
            try:
                self.num_workers: int | None = int(env_val)
            except Exception:
                self.num_workers = num_workers
        else:
            self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.dynamic_workers = dynamic_workers
        env_prio = os.getenv("RL_PRIORITIZED_REPLAY")
        if env_prio is not None:
            self.prioritized_replay = str(env_prio).lower() in {"1", "true", "yes"}
        else:
            self.prioritized_replay = bool(prioritized_replay)
        self.cpu_callback = cpu_callback
        self.dataset: _TradeDataset | None = None

    def recompute_workers(self, loader: DataLoader | None = None) -> None:
        """Recalculate ``num_workers`` and update ``loader`` in-place."""
        if (
            not self.dynamic_workers
            or self.dataset is None
            or self.num_workers is not None
        ):
            return
        new_val = _calc_num_workers(
            len(self.dataset),
            dynamic=True,
            cpu_callback=self.cpu_callback,
        )
        if loader is not None:
            loader.num_workers = new_val
            loader.pin_memory = self.pin_memory and new_val > 0
            loader.persistent_workers = self.persistent_workers and new_val > 0

    async def setup(self, stage: str | None = None) -> None:  # pragma: no cover - simple
        if self.mmap_path and Path(self.mmap_path).exists():
            mem = np.load(self.mmap_path, mmap_mode="r")
            snaps_arr = mem["snapshots"]
            trades_arr = mem["trades"]
            from types import SimpleNamespace
            from datetime import datetime

            trades = [
                SimpleNamespace(
                    token=str(r["token"]),
                    side=str(r["side"]),
                    price=float(r["price"]),
                    amount=float(r["amount"]),
                    timestamp=datetime.fromtimestamp(float(r["timestamp"])),
                )
                for r in trades_arr
            ]
            snaps = [
                SimpleNamespace(
                    token=str(r["token"]),
                    price=float(r["price"]),
                    depth=float(r["depth"]),
                    total_depth=float(r["total_depth"]),
                    slippage=float(r["slippage"]),
                    volume=float(r["volume"]),
                    imbalance=float(r["imbalance"]),
                    tx_rate=float(r["tx_rate"]),
                    whale_share=float(r["whale_share"]),
                    spread=float(r["spread"]),
                    sentiment=float(r["sentiment"]),
                    timestamp=datetime.fromtimestamp(float(r["timestamp"])),
                )
                for r in snaps_arr
            ]
        else:
            data = OfflineData(self.db_url)
            trades = await data.list_trades()
            snaps = await data.list_snapshots()
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
        num_workers = self.num_workers
        if num_workers is None:
            num_workers = _calc_num_workers(
                len(self.dataset),
                dynamic=self.dynamic_workers,
                cpu_callback=self.cpu_callback,
            )
        dataset = self.dataset
        sampler = None
        shuffle = True
        if self.prioritized_replay and WeightedRandomSampler is not None:
            wrapper = PrioritizedReplayDataset(dataset)
            dataset = wrapper
            sampler = wrapper.sampler
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.pin_memory and num_workers > 0,
            persistent_workers=self.persistent_workers and num_workers > 0,
        )


class LightningPPO(pl.LightningModule):
    """Minimal PPO actor-critic implementation."""

    def __init__(self, hidden_size: int = 32, lr: float = 3e-4, gamma: float = 0.99, clip_epsilon: float = 0.2) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.actor = nn.Sequential(
            nn.Linear(10, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(10, hidden_size),
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

    def configure_callbacks(self):  # pragma: no cover - simple
        return [_DynamicWorkersCallback()]


class LightningDQN(pl.LightningModule):
    """Simple DQN model trained via mean squared error."""

    def __init__(self, hidden_size: int = 32, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(10, hidden_size),
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

    def configure_callbacks(self):  # pragma: no cover - simple
        return [_DynamicWorkersCallback()]


class LightningA3C(pl.LightningModule):
    """Actor-critic model used for simple A3C training."""

    def __init__(self, hidden_size: int = 32, lr: float = 3e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.actor = nn.Sequential(
            nn.Linear(10, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(10, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.actor(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        states, actions, rewards = batch
        logits = self.actor(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        values = self.critic(states).squeeze()
        advantage = rewards - values.detach()
        actor_loss = -(log_probs[range(len(actions)), actions] * advantage).mean()
        critic_loss = self.loss_fn(values, rewards)
        loss = actor_loss + 0.5 * critic_loss
        self.log("loss", loss)
        self.log("reward", rewards.mean())
        return loss

    def configure_optimizers(self):  # pragma: no cover - simple
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        return torch.optim.Adam(params, lr=self.hparams.lr)

    def configure_callbacks(self):  # pragma: no cover - simple
        return [_DynamicWorkersCallback()]


class LightningDDPG(pl.LightningModule):
    """Minimal DDPG implementation."""

    def __init__(self, hidden_size: int = 32, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.actor = nn.Sequential(
            nn.Linear(10, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(11, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.actor(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int, optimizer_idx: int = 0):
        states, actions, rewards = batch
        if optimizer_idx == 0:
            actions = self.actor(states)
            q_vals = self.critic(torch.cat([states, actions], dim=1)).squeeze()
            loss = -q_vals.mean()
        else:
            act = actions.float().unsqueeze(1)
            q_pred = self.critic(torch.cat([states, act], dim=1)).squeeze()
            loss = self.loss_fn(q_pred, rewards)
        self.log("loss", loss)
        self.log("reward", rewards.mean())
        return loss

    def configure_optimizers(self):  # pragma: no cover - simple
        opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.hparams.lr)
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.hparams.lr)
        return [opt_a, opt_c]

    def configure_callbacks(self):  # pragma: no cover - simple
        return [_DynamicWorkersCallback()]


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


class _DynamicWorkersCallback(pl.callbacks.Callback):
    """Callback adjusting ``DataLoader`` worker count before each epoch."""

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # pragma: no cover - simple
        dm = getattr(trainer, "datamodule", None)
        if dm is None or not hasattr(dm, "recompute_workers"):
            return
        loaders = trainer.train_dataloader
        if isinstance(loaders, list):
            seq = loaders
        else:
            seq = [loaders]
        for loader in seq:
            dm.recompute_workers(loader)


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
        mmap_path: str | None = None,
        num_workers: int | None = None,
        device: str | None = None,
        metrics_url: str | None = None,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        dynamic_workers: bool | None = None,
        prioritized_replay: bool | None = None,
        cpu_callback: Callable[[], float] | None = None,
        worker_update_interval: float | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        dyn_workers = dynamic_workers
        env_dyn = os.getenv("RL_DYNAMIC_WORKERS")
        if env_dyn is not None:
            dyn_workers = str(env_dyn).lower() in {"1", "true", "yes"}
        prio_env = os.getenv("RL_PRIORITIZED_REPLAY")
        if prio_env is not None:
            prioritized_replay = str(prio_env).lower() in {"1", "true", "yes"}
        if mmap_path is None:
            default_mmap = Path("datasets/offline_data.npz")
            if not default_mmap.exists():
                _ensure_mmap_dataset(db_url, default_mmap)
            if default_mmap.exists():
                mmap_path = str(default_mmap)
        self.data = TradeDataModule(
            db_url,
            batch_size=batch_size,
            sims_per_token=sims_per_token,
            price_model_path=price_model_path,
            regime_weight=regime_weight,
            mmap_path=mmap_path,
            num_workers=num_workers,
            pin_memory=pin_memory if pin_memory is not None else True,
            persistent_workers=persistent_workers if persistent_workers is not None else True,
            dynamic_workers=dyn_workers,
            prioritized_replay=prioritized_replay,
            cpu_callback=cpu_callback,
        )
        self.worker_update_interval = worker_update_interval if worker_update_interval is not None else 10.0
        self._worker_last = 0.0
        self._worker_loaders: list[DataLoader] = []
        self._worker_sub = None
        if algo == "dqn":
            self.model: pl.LightningModule = LightningDQN()
        elif algo == "a3c":
            self.model = LightningA3C()
        elif algo == "ddpg":
            self.model = LightningDDPG()
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
        from .models import load_compiled_model
        self.jit_model = load_compiled_model(str(self.model_path), device)
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
        if self._worker_sub is not None:
            try:
                self._worker_sub.__exit__(None, None, None)
            except Exception:
                pass

    async def train(self) -> None:
        """Run one training cycle and persist weights."""
        await self.data.setup()
        self.trainer.fit(self.model, self.data)
        torch.save(self.model.state_dict(), self.model_path)
        from .models import export_torchscript, export_onnx

        try:
            export_torchscript(self.model.cpu(), self.model_path.with_suffix(".ptc"))
        except Exception as exc:  # pragma: no cover - scripting optional
            logging.getLogger(__name__).warning(
                "failed to export torchscript: %s", exc
            )
        if os.getenv("EXPORT_ONNX"):
            try:
                sample = torch.zeros(1, 9, dtype=torch.float32)
                export_onnx(self.model.cpu(), self.model_path.with_suffix(".onnx"), sample)
            except Exception as exc:  # pragma: no cover - optional
                logging.getLogger(__name__).warning("failed to export onnx: %s", exc)
        self.model.to(self.device)

    async def _loop(self, interval: float) -> None:
        while True:
            try:
                await self.train()
            except Exception as exc:  # pragma: no cover - logging
                self._logger.error("periodic training failed: %s", exc)
            await asyncio.sleep(interval)

    def start_periodic_retraining(self, interval: float = 3600.0) -> asyncio.Task:
        """Begin background task that periodically retrains the model."""
        if self._task is None:
            self._task = asyncio.create_task(self._loop(interval))
        return self._task

    def recompute_workers(self, loader: DataLoader | None = None) -> None:
        """Subscribe to metrics and adjust ``DataLoader`` workers on updates."""

        if not self.data.dynamic_workers or self.data.num_workers is not None:
            return

        if loader is not None and loader not in self._worker_loaders:
            self._worker_loaders.append(loader)

        if self._worker_sub is not None:
            return

        last = {"ts": 0.0}
        cpu_val = {"v": 0.0}

        def _handle(payload: Any) -> None:
            cpu = getattr(payload, "cpu", None)
            if isinstance(payload, dict):
                cpu = payload.get("cpu", cpu)
            if cpu is None:
                return
            try:
                cpu_val["v"] = float(cpu)
            except Exception:
                return
            now = time.time()
            if now - last["ts"] < self.worker_update_interval:
                return
            last["ts"] = now
            if self.data.dataset is None:
                return
            new_val = _calc_num_workers(
                len(self.data.dataset),
                dynamic=True,
                cpu_callback=lambda: cpu_val["v"],
            )
            for ld in list(self._worker_loaders):
                ld.num_workers = new_val
                ld.pin_memory = self.data.pin_memory and new_val > 0
                ld.persistent_workers = self.data.persistent_workers and new_val > 0

        sub = subscription("system_metrics_combined", _handle)
        sub.__enter__()
        self._worker_sub = sub


def fit(
    trades: Iterable[Any],
    snaps: Iterable[Any],
    *,
    model_path: str | Path = "ppo_model.pt",
    algo: str = "ppo",
    regime_weight: float = 1.0,
    device: str | None = None,
    dynamic_workers: bool = False,
    prioritized_replay: bool = False,
    cpu_callback: Callable[[], float] | None = None,
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
        ``"ppo"``, ``"dqn"``, ``"a3c"`` or ``"ddpg"`` model type.
    device:
        Optional accelerator string, ``"cuda"`` or ``"mps"``.
    """

    dataset = _TradeDataset(trades, snaps, regime_weight=regime_weight)
    if len(dataset) == 0:
        Path(model_path).touch()
        return

    if algo == "dqn":
        model: pl.LightningModule = LightningDQN()
    elif algo == "a3c":
        model = LightningA3C()
    elif algo == "ddpg":
        model = LightningDDPG()
    else:
        model = LightningPPO()

    use_compile = os.getenv("USE_TORCH_COMPILE", "1").lower() not in {"0", "false", "no"}
    if use_compile:
        try:
            if getattr(torch, "compile", None) and int(torch.__version__.split(".")[0]) >= 2:
                model = torch.compile(model)
        except Exception:
            pass

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
    num_workers = _calc_num_workers(
        len(dataset),
        dynamic=dynamic_workers,
        cpu_callback=cpu_callback,
    )
    sampler = None
    shuffle = True
    if prioritized_replay and WeightedRandomSampler is not None:
        wrapper = PrioritizedReplayDataset(dataset)
        dataset = wrapper
        sampler = wrapper.sampler
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        persistent_workers=num_workers > 0,
    )
    trainer.fit(model, train_dataloaders=loader)
    torch.save(model.state_dict(), path)
    from .models import export_torchscript, export_onnx

    try:
        export_torchscript(model.cpu(), path.with_suffix(".ptc"))
    except Exception as exc:  # pragma: no cover - scripting optional
        logging.getLogger(__name__).warning("failed to export torchscript: %s", exc)
    if os.getenv("EXPORT_ONNX"):
        try:
            sample = torch.zeros(1, 9, dtype=torch.float32)
            export_onnx(model.cpu(), path.with_suffix(".onnx"), sample)
        except Exception as exc:  # pragma: no cover - optional
            logging.getLogger(__name__).warning("failed to export onnx: %s", exc)
    model.to(device)


class MultiAgentRL:
    """Train and evaluate multiple RL models in parallel."""

    def __init__(
        self,
        *,
        db_url: str = "sqlite:///offline_data.db",
        algos: Iterable[str] | None = None,
        population_size: int = 2,
        model_base: str = "population_model.pt",
        device: str | None = None,
        regime_weight: float = 1.0,
        controller_path: str | Path | None = None,
        prioritized_replay: bool | None = None,
    ) -> None:
        self.db_url = db_url
        self.regime_weight = float(regime_weight)
        self.algos = list(algos or ["ppo"])
        self.population_size = int(population_size)
        self.device = device
        self.prioritized_replay = bool(prioritized_replay)
        ctrl_path = (
            Path(controller_path)
            if controller_path is not None
            else Path(model_base).with_name(f"{Path(model_base).stem}_controller.json")
        )
        self.trainers: list[RLTraining] = []
        self.model_paths: list[Path] = []
        self._algos_used: list[str] = []
        for i in range(self.population_size):
            algo = self.algos[i % len(self.algos)]
            path = Path(model_base).with_name(
                f"{Path(model_base).stem}_{i}_{algo}.pt"
            )
            self.model_paths.append(path)
            self.trainers.append(
                RLTraining(
                    db_url=db_url,
                    model_path=path,
                    algo=algo,
                    device=device,
                    regime_weight=self.regime_weight,
                    prioritized_replay=prioritized_replay,
                )
            )
            self._algos_used.append(algo)
        self.best_idx: int | None = None
        self.controller = PopulationRL(None, population_size=self.population_size, weights_path=str(ctrl_path))

    def close(self) -> None:
        for t in self.trainers:
            try:
                t.close()
            except Exception:
                pass

    def _score(self, model: pl.LightningModule, dataset: Dataset) -> float:
        loader = DataLoader(dataset, batch_size=64)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        total = 0.0
        with torch.no_grad():
            for states, actions, rewards in loader:
                states = states.to(device)
                if hasattr(model, "actor"):
                    out = model.actor(states)
                elif hasattr(model, "model"):
                    out = model.model(states)
                else:
                    out = model(states)
                if out.ndim == 2 and out.shape[1] == 1:
                    preds = (out.squeeze() >= 0).long().cpu()
                else:
                    preds = out.argmax(dim=1).cpu()
                mask = preds == actions
                if mask.any():
                    total += float(rewards[mask].sum())
        return total

    def train(self, trades: Iterable[Any], snaps: Iterable[Any]) -> None:
        dataset = _TradeDataset(trades, snaps, regime_weight=self.regime_weight)
        scores: list[float] = []
        for algo, trainer in zip(self._algos_used, self.trainers):
            fit(
                trades,
                snaps,
                model_path=trainer.model_path,
                algo=algo,
                regime_weight=self.regime_weight,
                device=self.device,
                prioritized_replay=self.prioritized_replay,
            )
            try:
                trainer.model.load_state_dict(torch.load(trainer.model_path))
            except Exception:
                pass
            scores.append(self._score(trainer.model, dataset))
        if scores:
            self.best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))

    def best_weights(self) -> dict[str, float]:
        if self.best_idx is None:
            return {}
        model = self.trainers[self.best_idx].model
        weights: dict[str, float] = {}
        try:
            for name, param in model.named_parameters():
                weights[name] = float(param.detach().mean())
        except Exception:
            pass
        return weights

    # ------------------------------------------------------------------
    def train_controller(self, agent_names: Iterable[str]) -> dict[str, float]:
        """Train the hierarchical controller and return best weights."""
        names = list(agent_names)
        if not any(cfg.get("weights") for cfg in self.controller.population):
            self.controller.population = [
                {"weights": {n: 1.0 for n in names}, "risk": {"risk_multiplier": 1.0}},
                {"weights": {n: 0.5 for n in names}, "risk": {"risk_multiplier": 1.0}},
            ]
        best = self.controller.evolve()
        w = best.get("weights", {}) if isinstance(best, dict) else {}
        return {n: float(w.get(n, 1.0)) for n in names}
