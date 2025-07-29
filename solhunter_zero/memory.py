from __future__ import annotations
import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from .base_memory import BaseMemory
from .event_bus import publish
from .schemas import TradeLogged

Base = declarative_base()

def utcnow():
    return datetime.datetime.utcnow()

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    token = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=utcnow)
    reason = Column(Text)


class VaRLog(Base):
    __tablename__ = "var_logs"

    id = Column(Integer, primary_key=True)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=utcnow)


class Memory(BaseMemory):
    def __init__(self, url: str = 'sqlite:///memory.db'):
        self.engine = create_engine(url, echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def log_trade(self, *, _broadcast: bool = True, **kwargs) -> int | None:
        with self.Session() as session:
            trade = Trade(**kwargs)
            session.add(trade)
            session.commit()
        if _broadcast:
            try:
                publish("trade_logged", TradeLogged(**kwargs))
            except Exception:
                pass
        return trade.id
    def log_var(self, value: float) -> None:
        """Record a value-at-risk measurement."""
        with self.Session() as session:
            rec = VaRLog(value=value)
            session.add(rec)
            session.commit()

    def list_trades(
        self,
        *,
        token: str | None = None,
        limit: int | None = None,
        since_id: int | None = None,
    ) -> list[Trade]:
        """Return trades optionally filtered by ``token`` or ``since_id``."""
        with self.Session() as session:
            q = session.query(Trade)
            if token is not None:
                q = q.filter_by(token=token)
            if since_id is not None:
                q = q.filter(Trade.id > since_id)
            q = q.order_by(Trade.id)
            if limit is not None:
                q = q.limit(limit)
            return list(q)

    def list_vars(self):
        with self.Session() as session:
            return session.query(VaRLog).all()
