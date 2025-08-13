from __future__ import annotations

import asyncio
import logging
import threading
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Any, Callable, Dict
import os
import time
import types
from sqlalchemy import select

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as exc:  # pragma: no cover - optional dependency

    class _TorchStub:
        class Tensor:
            pass

        class device:
            def __init__(self, *a, **k) -> None:
                pass

        class Module:
            def __init__(self, *a, **k) -> None:
                raise ImportError("torch is required for rl_daemon")

        def __getattr__(self, name):
            raise ImportError("torch is required for rl_daemon")

    class _DatasetStub:
        def __init__(self, *a, **k) -> None:
            raise ImportError("torch is required for rl_daemon")

    torch = nn = _TorchStub()  # type: ignore
    Dataset = DataLoader = _DatasetStub  # type: ignore

from .base_memory import BaseMemory
from .memory import Memory, Trade
from .offline_data import OfflineData, MarketSnapshot
from . import rl_training
from .rl_training import _ensure_mmap_dataset, MultiAgentRL
from .rl_algorithms import _A3C, _DDPG, TransformerPolicy
from .risk import average_correlation
from .portfolio import Portfolio
from .event_bus import (
    subscription,
    publish,
    send_heartbeat,
    connect_broker,
    _BROKER_URLS,  # type: ignore
)
from .config import get_broker_urls
from .schemas import ActionExecuted, RLCheckpoint, RLWeights
from .hierarchical_rl import (
    HighLevelPolicyNetwork,
    load_policy,
    save_policy,
    train_policy,
)
from .resource_monitor import get_cpu_usage
from .dynamic_limit import _step_limit
from .device import get_default_device
from .util import parse_bool_env

logger = logging.getLogger(__name__)


class _TradeDataset(Dataset):
    """Dataset wrapping trade history with risk and market metrics."""

    def __init__(
        self, trades: Iterable, snaps: Iterable, metrics: Tuple[float, float, float]
    ) -> None:
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


def portfolio_state(
    portfolio: Portfolio,
    token: str,
    price: float,
    *,
    depth: float = 0.0,
    tx_rate: float = 0.0,
) -> list[float]:
    """Return state vector for ``token`` using ``portfolio`` metrics."""

    pos = portfolio.balances.get(token)
    amount = float(pos.amount) if pos else 0.0

    seq = portfolio.price_history.get(token, [])
    if len(seq) >= 2:
        recent = seq[-30:]
        mean = sum(recent) / len(recent)
        var = sum((p - mean) ** 2 for p in recent) / len(recent)
        volatility = (var**0.5) / (mean + 1e-8)
        prev = seq[-2]
        trend = (seq[-1] - prev) / prev if prev else 0.0
    else:
        volatility = 0.0
        trend = 0.0

    prices = {t: h for t, h in portfolio.price_history.items() if len(h) >= 2}
    if token not in prices and seq:
        prices[token] = seq
    try:
        corr = average_correlation(prices) if prices else 0.0
    except Exception:
        corr = 0.0
    price_map = {t: vals[-1] for t, vals in prices.items()}
    price_map[token] = price
    drawdown = portfolio.current_drawdown(price_map)

    return [price, amount, drawdown, volatility, corr, tx_rate, trend, depth]


class _DQN(nn.Module):
    def __init__(self, input_size: int = 9, hidden_size: int = 32) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.model(x)


class _PPO(nn.Module):
    def __init__(
        self, input_size: int = 9, hidden_size: int = 32, clip_epsilon: float = 0.2
    ) -> None:
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


def _train_dqn(
    model: _DQN,
    data: Dataset,
    device: torch.device,
    *,
    dynamic_workers: bool = False,
    cpu_callback: Callable[[], float] | None = None,
) -> None:
    num_workers = rl_training._calc_num_workers(
        len(data), dynamic=dynamic_workers, cpu_callback=cpu_callback
    )
    loader = DataLoader(
        data,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        persistent_workers=num_workers > 0,
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


def _train_ppo(
    model: _PPO,
    data: Dataset,
    device: torch.device,
    *,
    dynamic_workers: bool = False,
    cpu_callback: Callable[[], float] | None = None,
) -> None:
    num_workers = rl_training._calc_num_workers(
        len(data), dynamic=dynamic_workers, cpu_callback=cpu_callback
    )
    loader = DataLoader(
        data,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        persistent_workers=num_workers > 0,
    )
    opt = torch.optim.Adam(
        list(model.actor.parameters()) + list(model.critic.parameters()), lr=3e-4
    )
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
            s2 = (
                torch.clamp(ratio, 1 - model.clip_epsilon, 1 + model.clip_epsilon)
                * advantages
            )
            actor_loss = -torch.min(s1, s2).mean()
            critic_loss = model.loss_fn(values, returns)
            loss = actor_loss + 0.5 * critic_loss
            opt.zero_grad()
            loss.backward()
            opt.step()


def _train_a3c(
    model: _A3C,
    data: Dataset,
    device: torch.device,
    *,
    dynamic_workers: bool = False,
    cpu_callback: Callable[[], float] | None = None,
) -> None:
    num_workers = rl_training._calc_num_workers(
        len(data), dynamic=dynamic_workers, cpu_callback=cpu_callback
    )
    loader = DataLoader(
        data,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        persistent_workers=num_workers > 0,
    )
    opt = torch.optim.Adam(
        list(model.actor.parameters()) + list(model.critic.parameters()), lr=3e-4
    )
    model.train()
    for _ in range(3):
        for states, actions, rewards in loader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            logits = model.actor(states)
            log_probs = torch.log_softmax(logits, dim=-1)
            values = model.critic(states).squeeze()
            advantage = rewards - values.detach()
            actor_loss = -(log_probs[range(len(actions)), actions] * advantage).mean()
            critic_loss = model.loss_fn(values, rewards)
            loss = actor_loss + 0.5 * critic_loss
            opt.zero_grad()
            loss.backward()
            opt.step()


def _train_ddpg(
    model: _DDPG,
    data: Dataset,
    device: torch.device,
    *,
    dynamic_workers: bool = False,
    cpu_callback: Callable[[], float] | None = None,
) -> None:
    num_workers = rl_training._calc_num_workers(
        len(data), dynamic=dynamic_workers, cpu_callback=cpu_callback
    )
    loader = DataLoader(
        data,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        persistent_workers=num_workers > 0,
    )
    actor_opt = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(model.critic.parameters(), lr=1e-3)
    model.train()
    for _ in range(3):
        for states, actions, rewards in loader:
            states = states.to(device)
            act = actions.float().unsqueeze(1).to(device)
            rewards = rewards.to(device)
            pred_q = model.critic(torch.cat([states, act], dim=1)).squeeze()
            critic_loss = model.loss_fn(pred_q, rewards)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            pred_act = model.actor(states)
            q_val = model.critic(torch.cat([states, pred_act], dim=1)).squeeze()
            actor_loss = -q_val.mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()


class RLDaemon:
    """Background trainer that updates RL models from trade history."""

    def __init__(
        self,
        memory: BaseMemory | None = None,
        memory_path: str = "sqlite:///memory.db",
        data_path: str = "offline_data.db",
        model_path: str | Path = "ppo_model.pt",
        algo: str = "ppo",
        policy: str = "mlp",
        device: str | None = None,
        *,
        agents: Iterable[Any] | None = None,
        queue: asyncio.Queue | None = None,
        metrics_url: str | None = None,
        dynamic_workers: bool | None = None,
        cpu_callback: Callable[[], float] | None = None,
        multi_rl: bool | None = None,
        rl_population_size: int = 2,
        live: bool | None = None,
        distributed_rl: bool | None = None,
        hierarchical_rl: bool | None = True,
        hierarchical_model_path: str | Path = "hier_policy.json",
        distributed_backend: str | None = None,
    ) -> None:
        self.memory = memory or Memory(memory_path)
        self.data_path = data_path
        self.data = OfflineData(f"sqlite:///{data_path}")
        _coro = _ensure_mmap_dataset(
            f"sqlite:///{data_path}", Path("datasets/offline_data.npz")
        )
        if _coro is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(_coro)
            else:  # pragma: no cover - existing loop
                loop.create_task(_coro)
        self.model_path = Path(model_path)
        self.algo = algo
        self.policy = policy
        self.multi_rl = bool(multi_rl)
        self.live = bool(live)
        self.distributed_rl = bool(distributed_rl)
        self.hierarchical_rl = (
            True if hierarchical_rl is None else bool(hierarchical_rl)
        )
        self.hierarchical_model_path = Path(hierarchical_model_path)
        self.distributed_backend = distributed_backend
        self.hier_policy = None
        self.hier_weights: Dict[str, float] = {}
        self.population: MultiAgentRL | None = None
        self.live_dataset: rl_training.LiveTradeDataset | None = None
        self.last_train_time: float | None = None
        self.checkpoint_path: str = str(self.model_path)
        self.device = get_default_device(device)
        device = self.device.type
        self.ray_trainer = None
        if self.distributed_backend == "ray":
            try:
                from .ray_training import RayTraining

                self.ray_trainer = RayTraining(
                    db_url=f"sqlite:///{data_path}",
                    model_path=self.model_path,
                    algo=algo,
                )
            except Exception as exc:
                logger.error("failed to initialize RayTraining: %s", exc)
        urls = get_broker_urls()
        if self.distributed_rl and urls and not _BROKER_URLS:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop:
                loop.create_task(connect_broker(urls))
            else:
                asyncio.run(connect_broker(urls))

        if self.live:
            self.live_dataset = rl_training.LiveTradeDataset(
                broker_url=urls[0] if self.distributed_rl and urls else None
            )
        if self.multi_rl:
            from .rl_training import MultiAgentRL

            self.population = MultiAgentRL(
                db_url=f"sqlite:///{data_path}",
                algos=[algo],
                population_size=rl_population_size,
                device=device,
            )
            self.model = _PPO()
        else:
            if algo == "dqn":
                self.model = _DQN()
            elif algo == "a3c":
                self.model = (
                    TransformerPolicy() if self.policy == "transformer" else _A3C()
                )
            elif algo == "ddpg":
                self.model = _DDPG()
            else:
                self.model = (
                    TransformerPolicy() if self.policy == "transformer" else _PPO()
                )

        use_compile = parse_bool_env("USE_TORCH_COMPILE", True)
        if use_compile:
            try:
                if (
                    getattr(torch, "compile", None)
                    and int(torch.__version__.split(".")[0]) >= 2
                ):
                    self.model = torch.compile(self.model)
            except Exception:
                pass
        if self.model_path.exists():
            try:
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device)
                )
            except Exception:  # pragma: no cover - corrupt file
                pass
        self.model.to(self.device)
        from .models import load_compiled_model

        ptc = self.model_path.with_suffix(".ptc")
        if ptc.exists():
            try:
                self.jit_model = torch.jit.load(os.fspath(ptc)).to(self.device)
            except Exception as exc:  # pragma: no cover - optional
                logger.error("failed to load compiled model: %s", exc)
                self.jit_model = load_compiled_model(str(self.model_path), self.device)
        else:
            self.jit_model = load_compiled_model(str(self.model_path), self.device)
        self.agents: List[Any] = list(agents) if agents else []
        if self.hierarchical_rl:
            names = [getattr(a, "name", str(i)) for i, a in enumerate(self.agents)]
            try:
                self.hier_policy = load_policy(
                    str(self.hierarchical_model_path), len(names)
                )
                self.hier_weights = self.hier_policy.predict(names)
            except Exception as exc:  # pragma: no cover - log errors
                logger.error("failed to load hierarchical policy: %s", exc)
                self.hier_weights = {}
        self._task: asyncio.Task | None = None
        self._hb_task: asyncio.Task | None = None
        self._proc: subprocess.Popen | None = None
        self._last_trade_id = 0
        self._last_snap_id = 0
        env_dyn = os.getenv("RL_DYNAMIC_WORKERS")
        self.dynamic_workers = (
            str(env_dyn).lower() in {"1", "true", "yes"}
            if env_dyn is not None
            else bool(dynamic_workers)
        )
        self._cpu_callback = cpu_callback
        self._cpu_usage = 0.0
        self._cpu_sub = None
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
        if self.dynamic_workers and cpu_callback is None:

            def _update_cpu(payload):
                try:
                    self._cpu_usage = float(
                        getattr(payload, "cpu", payload.get("cpu", payload))
                    )
                except Exception:
                    pass

            cpu_sub = subscription("system_metrics_combined", _update_cpu)
            cpu_sub.__enter__()
            self._cpu_sub = cpu_sub
            self._subscriptions.append(cpu_sub)
        from .metrics_client import start_metrics_exporter

        sub = start_metrics_exporter(metrics_url)
        self._subscriptions.append(
            types.SimpleNamespace(__exit__=lambda *a, **k: sub())
        )

        self._peer_weights: dict[str, float] = {}
        if self.distributed_rl:

            def _apply_weights(msg: Any) -> None:
                w = getattr(msg, "weights", None)
                if w is None and isinstance(msg, dict):
                    w = msg.get("weights")
                if isinstance(w, dict):
                    self._peer_weights = dict(w)

            sub = subscription("rl_weights", _apply_weights)
            sub.__enter__()
            self._subscriptions.append(sub)

    def close(self) -> None:
        for sub in self._subscriptions:
            sub.__exit__(None, None, None)
        if self.live_dataset is not None:
            self.live_dataset.close()
        if self._task:
            self._task.cancel()
        if self._hb_task:
            self._hb_task.cancel()
        if self.ray_trainer is not None:
            try:
                self.ray_trainer.close()
            except Exception:
                pass

    def _cpu(self) -> float:
        if self._cpu_callback:
            try:
                return float(self._cpu_callback())
            except Exception:
                return 0.0
        return self._cpu_usage

    async def _fetch_new(self) -> tuple[list[Trade], list[MarketSnapshot]]:
        """Return new trades and snapshots since the last training cycle."""
        if self.live_dataset is not None:
            return self.live_dataset.fetch_new()
        trades: list[Trade] = []
        snaps: list[MarketSnapshot] = []
        async with self.memory.Session() as session:
            q = (
                select(Trade)
                .filter(Trade.id > self._last_trade_id)
                .order_by(Trade.id)
            )
            result = await session.execute(q)
            trades = list(result.scalars().all())
            if trades:
                self._last_trade_id = trades[-1].id
        async with self.data.Session() as session:
            q = (
                select(MarketSnapshot)
                .filter(MarketSnapshot.id > self._last_snap_id)
                .order_by(MarketSnapshot.id)
            )
            result = await session.execute(q)
            snaps = list(result.scalars().all())
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

    async def predict_action(
        self,
        portfolio: Portfolio,
        token: str,
        price: float,
        *,
        depth: float = 0.0,
        tx_rate: float = 0.0,
    ) -> list[float]:
        """Return action logits predicted by the trained model."""

        state = portfolio_state(portfolio, token, price, depth=depth, tx_rate=tx_rate)
        tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
        model = self.jit_model or self.model
        model.eval()
        with torch.no_grad():
            if hasattr(model, "actor"):
                out = model.actor(tensor)
            elif hasattr(model, "model"):
                out = model.model(tensor)
            else:
                out = model(tensor)
        if hasattr(out, "squeeze"):
            out = out.squeeze()
        if hasattr(out, "cpu"):
            out = out.cpu()
        if hasattr(out, "tolist"):
            return list(out.tolist())
        if isinstance(out, (list, tuple)):
            return list(out)
        try:
            return [float(out)]
        except Exception:
            return []

    async def train(self) -> None:
        trades, snaps = await self._fetch_new()
        if not trades and not snaps:
            return
        if self.multi_rl and self.population is not None:
            self.population.train(trades, snaps)
            weights = self.population.best_weights()
        else:
            if self.distributed_backend == "ray" and self.ray_trainer is not None:
                self.ray_trainer.train()
            else:
                rl_training.fit(
                    trades,
                    snaps,
                    model_path=self.model_path,
                    algo=self.algo,
                    device=self.device.type,
                    dynamic_workers=self.dynamic_workers,
                    cpu_callback=self._cpu,
                )
            try:
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device)
                )
                from .models import export_torchscript, load_compiled_model

                try:
                    export_torchscript(
                        self.model.cpu(), self.model_path.with_suffix(".ptc")
                    )
                except Exception as exc:  # pragma: no cover - optional
                    logger.error("failed to export torchscript: %s", exc)
                finally:
                    self.model.to(self.device)
                self.jit_model = load_compiled_model(str(self.model_path), self.device)
            except Exception as exc:  # pragma: no cover - corrupt file
                logger.error("failed to load updated model: %s", exc)
            weights = {}
            try:
                for name, param in self.model.named_parameters():
                    weights[name] = float(param.detach().mean())
            except Exception:
                weights = {}
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
            RLWeights(weights=weights, risk={"risk_multiplier": self.current_risk}),
        )
        if self.hierarchical_rl:
            names = [getattr(a, "name", str(i)) for i, a in enumerate(self.agents)]
            if self.hier_policy is None or len(
                getattr(self.hier_policy, "weights", [])
            ) != len(names):
                self.hier_policy = load_policy(
                    str(self.hierarchical_model_path), len(names)
                )
            try:
                self.hier_weights = train_policy(self.hier_policy, trades, names)
                save_policy(self.hier_policy, str(self.hierarchical_model_path))
            except Exception as exc:  # pragma: no cover - log errors
                logger.error("failed to train hierarchical policy: %s", exc)
        else:
            self.hier_weights = {}
        reward = 0.0
        for t in trades:
            val = float(getattr(t, "amount", 0.0)) * float(getattr(t, "price", 0.0))
            if getattr(t, "direction", "buy") == "sell":
                reward += val
            else:
                reward -= val
        if self.distributed_rl:
            grads: dict[str, float] = {}
            try:
                for name, param in self.model.named_parameters():
                    grads[name] = float(param.detach().mean())
            except Exception:
                grads = {}
            publish("rl_metrics", {"loss": 0.0, "reward": reward, "gradients": grads})
        else:
            publish("rl_metrics", {"loss": 0.0, "reward": reward})

    async def _loop(self, interval: float) -> None:
        high = float(os.getenv("CPU_HIGH_THRESHOLD", "80") or 80)
        low = float(os.getenv("CPU_LOW_THRESHOLD", "40") or 40)
        kp = float(
            os.getenv("CONCURRENCY_SMOOTHING", os.getenv("CONCURRENCY_KP", "0.5"))
            or 0.5
        )
        ki = float(os.getenv("CONCURRENCY_KI", "0.0") or 0.0)
        min_i = float(
            os.getenv("RL_MIN_INTERVAL", str(max(0.01, interval / 2)))
            or max(0.01, interval / 2)
        )
        max_i = float(os.getenv("RL_MAX_INTERVAL", str(interval * 2)) or (interval * 2))
        delay = max(min_i, min(max_i, interval))
        while True:
            try:
                await self.train()
            except Exception as exc:  # pragma: no cover - log errors
                logger.error("daemon training failed: %s", exc)
            cpu = get_cpu_usage()
            if cpu > high:
                target = min(max_i, delay * 2)
            elif cpu < low:
                target = max(min_i, delay / 2)
            else:
                target = delay
            delay = float(
                _step_limit(int(delay), int(target), int(max_i), smoothing=kp, ki=ki)
            )
            delay = max(min_i, min(max_i, delay))
            await asyncio.sleep(delay)

    async def _watch_external(self, interval: float) -> None:
        high = float(os.getenv("CPU_HIGH_THRESHOLD", "80") or 80)
        low = float(os.getenv("CPU_LOW_THRESHOLD", "40") or 40)
        kp = float(
            os.getenv("CONCURRENCY_SMOOTHING", os.getenv("CONCURRENCY_KP", "0.5"))
            or 0.5
        )
        ki = float(os.getenv("CONCURRENCY_KI", "0.0") or 0.0)
        min_i = float(
            os.getenv("RL_MIN_INTERVAL", str(max(0.01, interval / 2)))
            or max(0.01, interval / 2)
        )
        max_i = float(os.getenv("RL_MAX_INTERVAL", str(interval * 2)) or (interval * 2))
        delay = max(min_i, min(max_i, interval))
        last = self.model_path.stat().st_mtime if self.model_path.exists() else 0.0
        while True:
            await asyncio.sleep(delay)
            try:
                mtime = self.model_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if mtime > last:
                last = mtime
                try:
                    self.model.load_state_dict(
                        torch.load(self.model_path, map_location=self.device)
                    )
                    from .models import export_torchscript, load_compiled_model

                    try:
                        export_torchscript(
                            self.model.cpu(), self.model_path.with_suffix(".ptc")
                        )
                    except Exception as exc:  # pragma: no cover - optional
                        logger.error("failed to export torchscript: %s", exc)
                    finally:
                        self.model.to(self.device)
                    self.jit_model = load_compiled_model(
                        str(self.model_path), self.device
                    )
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
            cpu = get_cpu_usage()
            if cpu > high:
                target = min(max_i, delay * 2)
            elif cpu < low:
                target = max(min_i, delay / 2)
            else:
                target = delay
            delay = float(
                _step_limit(int(delay), int(target), int(max_i), smoothing=kp, ki=ki)
            )
            delay = max(min_i, min(max_i, delay))

    def start(
        self,
        interval: float = 3600.0,
        *,
        auto_train: bool = False,
        tune_interval: float | None = None,
    ) -> asyncio.Task:
        """Begin the training loop in the current or a background event loop."""
        if self.live and interval == 3600.0:
            interval = 5.0
        if self._task is not None:
            return self._task

        # Auto-select accelerator if none was provided
        if self.device.type == "cpu":
            new_device = get_default_device()
            if new_device.type != "cpu":
                self.device = new_device
                self.model.to(self.device)
                for ag in self.agents:
                    try:
                        ag.device = self.device
                        if hasattr(ag, "model"):
                            ag.model.to(self.device)
                        if hasattr(ag, "actor"):
                            ag.actor.to(self.device)
                        if hasattr(ag, "critic"):
                            ag.critic.to(self.device)
                    except Exception:
                        continue

        if auto_train:
            if tune_interval is None:
                tune_interval = interval
            script = (
                Path(__file__).resolve().parent.parent / "scripts" / "auto_train_rl.py"
            )
            self._proc = subprocess.Popen(
                [
                    sys.executable,
                    str(script),
                    "--db",
                    self.data_path,
                    "--model",
                    str(self.model_path),
                    "--interval",
                    str(tune_interval),
                ],
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
            if self._hb_task is None:
                self._hb_task = loop.create_task(send_heartbeat("rl_daemon"))
            return self._task

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - no running loop
            loop = asyncio.new_event_loop()
            t = threading.Thread(target=loop.run_forever, daemon=True)
            t.start()
        if self.dynamic_workers and self._cpu_callback is None:
            from . import resource_monitor

            loop.call_soon_threadsafe(resource_monitor.start_monitor)
        self._task = loop.create_task(self._loop(interval))
        if self._hb_task is None:
            self._hb_task = loop.create_task(send_heartbeat("rl_daemon"))
        return self._task


def parameter_server() -> Any:
    """Aggregate gradient updates from ``rl_metrics`` events and publish weights."""

    updates: list[dict[str, float]] = []
    weights: dict[str, float] = {}

    def _on_metrics(msg: Any) -> None:
        grads = getattr(msg, "gradients", None)
        if grads is None and isinstance(msg, dict):
            grads = msg.get("gradients")
        if not isinstance(grads, dict):
            return
        updates.append({k: float(v) for k, v in grads.items()})
        count = len(updates)
        agg: dict[str, float] = {}
        for g in updates:
            for k, v in g.items():
                agg[k] = agg.get(k, 0.0) + v
        for k, v in agg.items():
            weights[k] = v / count
        publish("rl_weights", RLWeights(weights=weights))

    sub = subscription("rl_metrics", _on_metrics)
    sub.__enter__()
    return sub
