from __future__ import annotations

import datetime
import os
import asyncio
from contextlib import suppress
from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
    select,
    insert,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
)

Base = declarative_base()


def utcnow() -> datetime.datetime:
    return datetime.datetime.utcnow()


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True)
    token = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    depth = Column(Float, nullable=False)
    total_depth = Column(Float, nullable=False, default=0.0)
    slippage = Column(Float, nullable=False, default=0.0)
    volume = Column(Float, nullable=False, default=0.0)
    imbalance = Column(Float, nullable=False)
    tx_rate = Column(Float, nullable=False, default=0.0)
    whale_share = Column(Float, nullable=False, default=0.0)
    spread = Column(Float, nullable=False, default=0.0)
    sentiment = Column(Float, nullable=False, default=0.0)
    timestamp = Column(DateTime, default=utcnow)


class MarketTrade(Base):
    __tablename__ = "market_trades"

    id = Column(Integer, primary_key=True)
    token = Column(String, nullable=False)
    side = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=utcnow)


class OfflineData:
    """Store order book snapshots and trade metrics for offline training."""

    def __init__(self, url: str = "sqlite:///offline_data.db") -> None:
        if url.startswith("sqlite:///") and "+" not in url:
            url = url.replace("sqlite:///", "sqlite+aiosqlite:///")
        self.engine = create_async_engine(url, echo=False, future=True)
        self.Session = async_sessionmaker(bind=self.engine, expire_on_commit=False)
        self._queue: asyncio.Queue[tuple[str, dict]] | None = None
        self._writer_task: asyncio.Task | None = None
        self._batch_size = 100
        self._interval = 1.0
        self._flush_max_batch: int | None = None
        import asyncio
        async def _init_models():
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
        if loop.is_running():
            self._init_task = loop.create_task(_init_models())
        else:
            loop.run_until_complete(_init_models())
            self._init_task = loop.create_future()
            self._init_task.set_result(None)

    def start_writer(
        self,
        batch_size: int = 100,
        interval: float = 1.0,
        max_batch: int | None = None,
    ) -> None:
        """Start background writer flushing queued entries."""
        env_batch = os.getenv("OFFLINE_BATCH_SIZE")
        env_interval = os.getenv("OFFLINE_FLUSH_INTERVAL")
        env_max = os.getenv("OFFLINE_FLUSH_MAX_BATCH")
        if env_batch is not None:
            batch_size = int(env_batch)
        if env_interval is not None:
            interval = float(env_interval)
        if env_max is not None:
            max_batch = int(env_max)
        self._batch_size = batch_size
        self._interval = interval
        self._flush_max_batch = max_batch
        self._queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        self._writer_task = loop.create_task(self._writer())

    async def _writer(self) -> None:
        assert self._queue is not None
        pending: list[tuple[str, dict]] = []
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=self._interval
                    )
                    pending.append(item)
                    if len(pending) >= self._batch_size:
                        await self._flush(pending)
                        pending.clear()
                except asyncio.TimeoutError:
                    if pending:
                        await self._flush(pending)
                        pending.clear()
        except asyncio.CancelledError:
            pass
        finally:
            if pending:
                await self._flush(pending)
            # drain remaining items
            while self._queue and not self._queue.empty():
                pending.append(self._queue.get_nowait())
                self._queue.task_done()
            if pending:
                await self._flush(pending)

    async def _flush(self, items: list[tuple[str, dict]]) -> None:
        max_batch = self._flush_max_batch or len(items)
        async with self.Session() as session:
            for i in range(0, len(items), max_batch):
                batch = items[i : i + max_batch]
                snaps = [d for t, d in batch if t == "snap"]
                trades = [d for t, d in batch if t != "snap"]
                if snaps:
                    await session.execute(insert(MarketSnapshot), snaps)
                if trades:
                    await session.execute(insert(MarketTrade), trades)
            await session.commit()

    async def log_snapshot(
        self,
        token: str,
        price: float,
        depth: float,
        imbalance: float,
        total_depth: float = 0.0,
        slippage: float = 0.0,
        volume: float = 0.0,
        tx_rate: float = 0.0,
        whale_share: float = 0.0,
        spread: float = 0.0,
        sentiment: float = 0.0,
    ) -> None:
        await self._init_task
        data = dict(
            token=token,
            price=price,
            depth=depth,
            total_depth=total_depth,
            slippage=slippage,
            volume=volume,
            imbalance=imbalance,
            tx_rate=tx_rate,
            whale_share=whale_share,
            spread=spread,
            sentiment=sentiment,
        )
        if self._queue is not None:
            await self._queue.put(("snap", data))
        else:
            async with self.Session() as session:
                session.add(MarketSnapshot(**data))
                await session.commit()

    async def log_trade(
        self,
        token: str,
        side: str,
        price: float,
        amount: float,
    ) -> None:
        """Record an executed trade for offline learning."""
        await self._init_task
        data = dict(token=token, side=side, price=price, amount=amount)
        if self._queue is not None:
            await self._queue.put(("trade", data))
        else:
            async with self.Session() as session:
                session.add(MarketTrade(**data))
                await session.commit()

    async def list_snapshots(self, token: str | None = None):
        await self._init_task
        async with self.Session() as session:
            q = select(MarketSnapshot)
            if token:
                q = q.filter(MarketSnapshot.token == token)
            q = q.order_by(MarketSnapshot.timestamp)
            result = await session.execute(q)
            return list(result.scalars().all())

    async def list_trades(self, token: str | None = None):
        await self._init_task
        async with self.Session() as session:
            q = select(MarketTrade)
            if token:
                q = q.filter(MarketTrade.token == token)
            q = q.order_by(MarketTrade.timestamp)
            result = await session.execute(q)
            return list(result.scalars().all())

    async def export_npz(self, out_path: str, token: str | None = None):
        """Export snapshots and trades to a compressed ``.npz`` file.

        Parameters
        ----------
        out_path:
            Destination ``.npz`` file.
        token:
            Optional token filter.
        """
        import numpy as np

        snaps = await self.list_snapshots(token)
        trades = await self.list_trades(token)

        snap_dtype = [
            ("token", "U32"),
            ("price", "f4"),
            ("depth", "f4"),
            ("total_depth", "f4"),
            ("slippage", "f4"),
            ("volume", "f4"),
            ("imbalance", "f4"),
            ("tx_rate", "f4"),
            ("whale_share", "f4"),
            ("spread", "f4"),
            ("sentiment", "f4"),
            ("timestamp", "f8"),
        ]

        snap_iter = (
            (
                s.token,
                float(s.price),
                float(s.depth),
                float(getattr(s, "total_depth", 0.0)),
                float(getattr(s, "slippage", 0.0)),
                float(getattr(s, "volume", 0.0)),
                float(s.imbalance),
                float(getattr(s, "tx_rate", 0.0)),
                float(getattr(s, "whale_share", 0.0)),
                float(getattr(s, "spread", 0.0)),
                float(getattr(s, "sentiment", 0.0)),
                s.timestamp.timestamp(),
            )
            for s in snaps
        )

        snap_arr = np.fromiter(snap_iter, dtype=snap_dtype, count=len(snaps))

        trade_dtype = [
            ("token", "U32"),
            ("side", "U8"),
            ("price", "f4"),
            ("amount", "f4"),
            ("timestamp", "f8"),
        ]

        trade_iter = (
            (
                t.token,
                t.side,
                float(t.price),
                float(t.amount),
                t.timestamp.timestamp(),
            )
            for t in trades
        )

        trade_arr = np.fromiter(trade_iter, dtype=trade_dtype, count=len(trades))

        np.savez_compressed(out_path, snapshots=snap_arr, trades=trade_arr)
        return np.load(out_path, mmap_mode="r")

    async def close(self) -> None:
        """Flush pending items and dispose engine."""
        await self._init_task
        if self._writer_task:
            self._writer_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._writer_task
            self._writer_task = None
        if self._queue and not self._queue.empty():
            await self._flush([self._queue.get_nowait() for _ in range(self._queue.qsize())])
        await self.engine.dispose()
