from __future__ import annotations

import datetime
from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
    select,
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
        import asyncio
        async def _init_models():
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            self._init_task = loop.create_task(_init_models())
        else:
            loop.run_until_complete(_init_models())
            self._init_task = loop.create_future()
            self._init_task.set_result(None)

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
        async with self.Session() as session:
            snap = MarketSnapshot(
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
            session.add(snap)
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
        async with self.Session() as session:
            trade = MarketTrade(
                token=token,
                side=side,
                price=price,
                amount=amount,
            )
            session.add(trade)
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

        snap_arr = np.array(
            [
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
            ],
            dtype=
            [
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
            ],
        )

        trade_arr = np.array(
            [
                (
                    t.token,
                    t.side,
                    float(t.price),
                    float(t.amount),
                    t.timestamp.timestamp(),
                )
                for t in trades
            ],
            dtype=[
                ("token", "U32"),
                ("side", "U8"),
                ("price", "f4"),
                ("amount", "f4"),
                ("timestamp", "f8"),
            ],
        )

        np.savez_compressed(out_path, snapshots=snap_arr, trades=trade_arr)
        return np.load(out_path, mmap_mode="r")
