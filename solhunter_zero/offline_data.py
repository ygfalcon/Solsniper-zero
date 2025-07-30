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
        self.engine = create_async_engine(url, echo=False, future=True)
        self.Session = async_sessionmaker(bind=self.engine, expire_on_commit=False)
        import asyncio
        async def _init_models():
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_init_models())
        else:
            loop.run_until_complete(_init_models())

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
        async with self.Session() as session:
            q = select(MarketSnapshot)
            if token:
                q = q.filter(MarketSnapshot.token == token)
            q = q.order_by(MarketSnapshot.timestamp)
            result = await session.execute(q)
            return list(result.scalars().all())

    async def list_trades(self, token: str | None = None):
        async with self.Session() as session:
            q = select(MarketTrade)
            if token:
                q = q.filter(MarketTrade.token == token)
            q = q.order_by(MarketTrade.timestamp)
            result = await session.execute(q)
            return list(result.scalars().all())
