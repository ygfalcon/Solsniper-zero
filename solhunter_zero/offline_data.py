from __future__ import annotations

import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
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
    imbalance = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=utcnow)


class OfflineData:
    """Store order book snapshots and trade metrics for offline training."""

    def __init__(self, url: str = "sqlite:///offline_data.db") -> None:
        self.engine = create_engine(url, echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def log_snapshot(
        self, token: str, price: float, depth: float, imbalance: float
    ) -> None:
        with self.Session() as session:
            snap = MarketSnapshot(
                token=token,
                price=price,
                depth=depth,
                imbalance=imbalance,
            )
            session.add(snap)
            session.commit()

    def list_snapshots(self, token: str | None = None):
        with self.Session() as session:
            q = session.query(MarketSnapshot)
            if token:
                q = q.filter_by(token=token)
            return list(q.order_by(MarketSnapshot.timestamp))
