from __future__ import annotations
import datetime
import asyncio
import threading
import atexit
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
    def __init__(
        self,
        url: str = "sqlite:///memory.db",
        *,
        commit_interval: float | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.engine = create_engine(url, echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

        self.commit_interval = commit_interval
        self.batch_size = batch_size
        self._trade_buffer: list[tuple[Trade, bool]] = []
        self._var_buffer: list[VaRLog] = []
        self._flush_lock = threading.Lock()
        self._flush_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        if commit_interval:
            self._start_flush_loop()
        atexit.register(self.close)

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _start_flush_loop(self) -> None:
        """Launch background task that flushes the buffer periodically."""
        try:
            self._loop = asyncio.get_running_loop()
            self._flush_task = self._loop.create_task(self._periodic_flush())
        except RuntimeError:
            self._loop = asyncio.new_event_loop()

            def _run() -> None:
                asyncio.set_event_loop(self._loop)
                self._flush_task = self._loop.create_task(self._periodic_flush())
                self._loop.run_forever()

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            self._thread = thread

    # ------------------------------------------------------------------
    async def _periodic_flush(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.commit_interval)
                self.flush()
        except asyncio.CancelledError:  # pragma: no cover - cancelled during shutdown
            pass

    # ------------------------------------------------------------------
    def flush(self) -> None:
        """Write any buffered rows to the database."""
        with self._flush_lock:
            if not self._trade_buffer and not self._var_buffer:
                return
            trades = [t for t, _ in self._trade_buffer]
            broadcasts = [b for _, b in self._trade_buffer]
            vars_ = list(self._var_buffer)
            self._trade_buffer.clear()
            self._var_buffer.clear()
        with self.Session() as session:
            if trades:
                session.add_all(trades)
            if vars_:
                session.add_all(vars_)
            session.commit()
            committed_trades = list(trades)
        for trade, do_broadcast in zip(committed_trades, broadcasts):
            if do_broadcast:
                try:
                    publish(
                        "trade_logged",
                        TradeLogged(
                            token=trade.token,
                            direction=trade.direction,
                            amount=float(trade.amount),
                            price=float(trade.price),
                            reason=trade.reason,
                            trade_id=trade.id,
                        ),
                    )
                except Exception:
                    pass

    def log_trade(self, *, _broadcast: bool = True, **kwargs) -> int | None:
        """Record a trade and return its id when committed."""
        if self.commit_interval or self.batch_size:
            trade = Trade(**kwargs)
            with self._flush_lock:
                self._trade_buffer.append((trade, _broadcast))
                should_flush = self.batch_size and len(self._trade_buffer) >= self.batch_size
            if should_flush:
                self.flush()
            return None

        with self.Session() as session:
            trade = Trade(**kwargs)
            session.add(trade)
            session.commit()
        if _broadcast:
            try:
                publish(
                    "trade_logged",
                    TradeLogged(
                        token=kwargs.get("token"),
                        direction=kwargs.get("direction"),
                        amount=float(kwargs.get("amount", 0.0)),
                        price=float(kwargs.get("price", 0.0)),
                        reason=kwargs.get("reason"),
                        trade_id=trade.id,
                    ),
                )
            except Exception:
                pass
        return trade.id
    def log_var(self, value: float) -> None:
        """Record a value-at-risk measurement."""
        if self.commit_interval or self.batch_size:
            with self._flush_lock:
                self._var_buffer.append(VaRLog(value=value))
                should_flush = self.batch_size and len(self._var_buffer) >= self.batch_size
            if should_flush:
                self.flush()
            return

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

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Flush pending rows and cancel background tasks."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            if self._loop:
                try:
                    self._loop.call_soon_threadsafe(lambda: None)
                except RuntimeError:  # pragma: no cover - loop closed
                    pass
        if getattr(self, "_thread", None):
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
        self.flush()
