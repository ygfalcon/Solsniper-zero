from __future__ import annotations
import datetime
from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Text,
    select,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)

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
        if url.startswith('sqlite:///'):
            url = url.replace('sqlite://', 'sqlite+aiosqlite://', 1)
        self.engine = create_async_engine(url, echo=False, future=True)
        self.Session = async_sessionmaker(bind=self.engine, expire_on_commit=False)
        import asyncio
        async def _init_models():
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            loop.create_task(_init_models())
        else:
            loop.run_until_complete(_init_models())

    async def log_trade(self, *, _broadcast: bool = True, **kwargs) -> int | None:
        async with self.Session() as session:
            trade = Trade(**kwargs)
            session.add(trade)
            await session.commit()
        if _broadcast:
            try:
                publish("trade_logged", TradeLogged(**kwargs))
            except Exception:
                pass
        return trade.id
    async def _log_var_async(self, value: float) -> None:
        async with self.Session() as session:
            rec = VaRLog(value=value)
            session.add(rec)
            await session.commit()

    def log_var(self, value: float) -> None:
        """Record a value-at-risk measurement."""
        import asyncio
        asyncio.run(self._log_var_async(value))

    async def list_trades(
        self,
        *,
        token: str | None = None,
        limit: int | None = None,
        since_id: int | None = None,
    ) -> list[Trade]:
        """Return trades optionally filtered by ``token`` or ``since_id``."""
        async with self.Session() as session:
            q = select(Trade)
            if token is not None:
                q = q.filter(Trade.token == token)
            if since_id is not None:
                q = q.filter(Trade.id > since_id)
            q = q.order_by(Trade.id)
            if limit is not None:
                q = q.limit(limit)
            result = await session.execute(q)
            return list(result.scalars().all())

    async def _list_vars_async(self):
        async with self.Session() as session:
            result = await session.execute(select(VaRLog))
            return list(result.scalars().all())

    def list_vars(self):
        import asyncio
        return asyncio.run(self._list_vars_async())
