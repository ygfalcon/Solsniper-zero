from __future__ import annotations

import asyncio
import logging
import os
import datetime
from pathlib import Path
from typing import Any, Iterable, List, Tuple
import time
import psutil

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
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
from .regime import detect_regime
from .event_bus import publish, subscription


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
        start_time = time.time()
        self.price_model_path = price_model_path
        self.regime_weight = float(regime_weight)

        trade_list = list(trades)
        n = len(trade_list)

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

        snap_map: dict[str, List[Any]] = {}
        for s in snaps:
            snap_map.setdefault(s.token, []).append(s)

        snap_data: dict[str, dict[str, np.ndarray]] = {}
        for token, seq in snap_map.items():
            seq.sort(key=lambda x: x.timestamp)
            snap_data[token] = {
                "ts": np.array([s.timestamp.timestamp() for s in seq], dtype=np.float64),
                "depth": np.array([float(getattr(s, "depth", 0.0)) for s in seq], dtype=np.float32),
                "slippage": np.array(
                    [float(getattr(s, "slippage", 0.0)) for s in seq], dtype=np.float32
                ),
                "imbalance": np.array(
                    [float(getattr(s, "imbalance", 0.0)) for s in seq], dtype=np.float32
                ),
                "tx_rate": np.array(
                    [float(getattr(s, "tx_rate", 0.0)) for s in seq], dtype=np.float32
                ),
            }

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
        if snap_data:
            uniq_tokens = np.unique(tokens)

            snap_tokens = []
            snap_ts = []
            snap_depth = []
            snap_slip = []
            snap_imb = []
            snap_tx = []
            for tok in uniq_tokens:
                data = snap_data.get(tok)
                if data is None:
                    # token may have no snapshot data
                    continue
                m = np.full_like(data["ts"], fill_value=tok, dtype=object)
                snap_tokens.append(m)
                snap_ts.append(data["ts"])
                snap_depth.append(data["depth"])
                snap_slip.append(data["slippage"])
                snap_imb.append(data["imbalance"])
                snap_tx.append(data["tx_rate"])

            if snap_ts:
                snap_tokens = np.concatenate(snap_tokens)
                snap_ts = np.concatenate(snap_ts)
                snap_depth = np.concatenate(snap_depth)
                snap_slip = np.concatenate(snap_slip)
                snap_imb = np.concatenate(snap_imb)
                snap_tx = np.concatenate(snap_tx)

                token_map = {t: i for i, t in enumerate(np.unique(snap_tokens))}
                snap_tok_idx = np.array([token_map[t] for t in snap_tokens])
                trade_tok_idx = np.array([token_map.get(t, -1) for t in tokens])

                shift = np.max(snap_ts) + 1.0
                encoded_snaps = snap_tok_idx * shift + snap_ts
                order = np.argsort(encoded_snaps)
                encoded_snaps = encoded_snaps[order]
                snap_depth = snap_depth[order]
                snap_slip = snap_slip[order]
                snap_imb = snap_imb[order]
                snap_tx = snap_tx[order]

                encoded_trades = trade_tok_idx * shift + timestamps
                idx = np.searchsorted(encoded_snaps, encoded_trades, side="right") - 1
                idx[idx < 0] = 0
                valid = trade_tok_idx >= 0
                depth[valid] = snap_depth[idx[valid]]
                slippage[valid] = snap_slip[idx[valid]]
                imbalance[valid] = snap_imb[idx[valid]]
                tx_rate[valid] = snap_tx[idx[valid]]

        unique_tokens = np.unique(tokens)
        pred_cache: dict[str, float] = {}
        for tok in unique_tokens:
            try:
                pred_cache[tok] = predict_price_movement(tok, model_path=price_model_path)
            except Exception:
                pred_cache[tok] = 0.0
        preds = np.array([pred_cache.get(tok, 0.0) for tok in tokens], dtype=np.float32)

        price_hist: dict[str, List[float]] = {}
        regimes = np.zeros(n, dtype=np.float32)
        for i, (tok, p) in enumerate(zip(tokens, prices)):
            hist = price_hist.setdefault(tok, [])
            regime = detect_regime(hist)
            regimes[i] = {
                "bull": 1.0,
                "bear": -1.0,
            }.get(regime, 0.0) * self.regime_weight
            hist.append(float(p))

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

        self.states = states
        self.actions = actions
        self.rewards = rewards
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
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
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

    def configure_callbacks(self):  # pragma: no cover - simple
        return [_DynamicWorkersCallback()]


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
        cpu_callback: Callable[[], float] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        dyn_workers = dynamic_workers
        env_dyn = os.getenv("RL_DYNAMIC_WORKERS")
        if env_dyn is not None:
            dyn_workers = str(env_dyn).lower() in {"1", "true", "yes"}
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
            cpu_callback=cpu_callback,
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

    async def train(self) -> None:
        """Run one training cycle and persist weights."""
        await self.data.setup()
        self.trainer.fit(self.model, self.data)
        torch.save(self.model.state_dict(), self.model_path)
        script_path = self.model_path.with_suffix(".ptc")
        try:
            scripted = torch.jit.script(self.model.cpu())
            scripted.save(script_path)
        except Exception as exc:  # pragma: no cover - scripting optional
            logging.getLogger(__name__).warning(
                "failed to script model: %s", exc
            )
        finally:
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


def fit(
    trades: Iterable[Any],
    snaps: Iterable[Any],
    *,
    model_path: str | Path = "ppo_model.pt",
    algo: str = "ppo",
    regime_weight: float = 1.0,
    device: str | None = None,
    dynamic_workers: bool = False,
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
    num_workers = _calc_num_workers(
        len(dataset),
        dynamic=dynamic_workers,
        cpu_callback=cpu_callback,
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        persistent_workers=num_workers > 0,
    )
    trainer.fit(model, train_dataloaders=loader)
    torch.save(model.state_dict(), path)
    script_path = path.with_suffix(".ptc")
    try:
        scripted = torch.jit.script(model.cpu())
        scripted.save(script_path)
    except Exception as exc:  # pragma: no cover - scripting optional
        logging.getLogger(__name__).warning(
            "failed to script model: %s", exc
        )
    finally:
        model.to(device)
