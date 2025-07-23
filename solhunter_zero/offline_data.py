from __future__ import annotations

import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker

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
        self.engine = create_engine(url, echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def log_snapshot(
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
    ) -> None:
        with self.Session() as session:
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
            )
            session.add(snap)
            session.commit()

    def log_trade(
        self,
        token: str,
        side: str,
        price: float,
        amount: float,
    ) -> None:
        """Record an executed trade for offline learning."""
        with self.Session() as session:
            trade = MarketTrade(
                token=token,
                side=side,
                price=price,
                amount=amount,
            )
            session.add(trade)
            session.commit()

    def list_snapshots(self, token: str | None = None):
        with self.Session() as session:
            q = session.query(MarketSnapshot)
            if token:
                q = q.filter_by(token=token)
            return list(q.order_by(MarketSnapshot.timestamp))

    def list_trades(self, token: str | None = None):
        with self.Session() as session:
            q = session.query(MarketTrade)
            if token:
                q = q.filter_by(token=token)
            return list(q.order_by(MarketTrade.timestamp))
