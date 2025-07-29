from __future__ import annotations

import os
import datetime
import uuid as uuid_module
from typing import List, Any

import numpy as np
try:  # optional heavy deps
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    faiss = None
    SentenceTransformer = None
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from .base_memory import BaseMemory
from .event_bus import publish, subscription
from .schemas import TradeLogged


Base = declarative_base()


def utcnow() -> datetime.datetime:
    return datetime.datetime.utcnow()


class SimulationSummary(Base):
    __tablename__ = "simulation_summaries"

    id = Column(Integer, primary_key=True)
    token = Column(String, nullable=False)
    agent = Column(String)
    expected_roi = Column(Float, nullable=False)
    success_prob = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=utcnow)


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    uuid = Column(String, unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    token = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=utcnow)
    reason = Column(Text)
    context = Column(Text)
    emotion = Column(String)
    simulation_id = Column(Integer, ForeignKey("simulation_summaries.id"))


class AdvancedMemory(BaseMemory):
    """Store trades with semantic search on context text."""

    def __init__(
        self,
        url: str = "sqlite:///memory.db",
        index_path: str = "trade.index",
        replicate: bool = False,
    ) -> None:
        self.engine = create_engine(url, echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

        self.index_path = index_path
        if faiss is not None and SentenceTransformer is not None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            dim = self.model.get_sentence_embedding_dimension()
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
            else:
                self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
        else:  # fallback without embeddings
            self.model = None
            self.index = None

        self._replication_sub = None
        if replicate:
            self._replication_sub = subscription("trade_logged", self._apply_remote)
            self._replication_sub.__enter__()

    # ------------------------------------------------------------------
    def _add_embedding(self, text: str, trade_id: int) -> None:
        if self.index is None or self.model is None:
            return
        vec = self.model.encode([text])[0].astype("float32")
        self.index.add_with_ids(
            np.array([vec]), np.array([trade_id], dtype="int64")
        )
        faiss.write_index(self.index, self.index_path)

    # ------------------------------------------------------------------
    def _apply_remote(self, msg: Any) -> None:
        data = msg if isinstance(msg, dict) else msg.__dict__
        trade_uuid = data.get("uuid")
        if trade_uuid is not None:
            with self.Session() as session:
                exists = session.query(Trade).filter_by(uuid=trade_uuid).first()
                if exists:
                    return
        data.pop("trade_id", None)
        self.log_trade(_broadcast=False, **data)

    # ------------------------------------------------------------------
    def log_simulation(
        self,
        token: str,
        *,
        expected_roi: float,
        success_prob: float,
        agent: str | None = None,
    ) -> int:
        """Insert a simulation summary and return its id."""
        with self.Session() as session:
            sim = SimulationSummary(
                token=token,
                agent=agent,
                expected_roi=expected_roi,
                success_prob=success_prob,
            )
            session.add(sim)
            session.commit()
            return sim.id

    # ------------------------------------------------------------------
    def log_trade(
        self,
        *,
        token: str,
        direction: str,
        amount: float,
        price: float,
        uuid: str | None = None,
        reason: str | None = None,
        context: str = "",
        emotion: str = "",
        simulation_id: int | None = None,
        _broadcast: bool = True,
    ) -> int:
        with self.Session() as session:
            trade_uuid = uuid or str(uuid_module.uuid4())
            trade = Trade(
                token=token,
                direction=direction,
                amount=amount,
                price=price,
                uuid=trade_uuid,
                reason=reason,
                context=context,
                emotion=emotion,
                simulation_id=simulation_id,
            )
            session.add(trade)
            session.commit()
            text = context or f"{direction} {token}"
            self._add_embedding(text, trade.id)
        if _broadcast:
            try:
                publish(
                    "trade_logged",
                    TradeLogged(
                        token=token,
                        direction=direction,
                        amount=amount,
                        price=price,
                        reason=reason,
                        context=context,
                        emotion=emotion,
                        simulation_id=simulation_id,
                        uuid=trade_uuid,
                        trade_id=trade.id,
                    ),
                )
            except Exception:
                pass
        return trade.id

    # ------------------------------------------------------------------
    def list_trades(
        self,
        *,
        token: str | None = None,
        limit: int | None = None,
        since_id: int | None = None,
    ) -> List[Trade]:
        """Return trades optionally filtered by token or id."""
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

    # ------------------------------------------------------------------
    def simulation_success_rate(self, token: str, *, agent: str | None = None) -> float:
        """Return the average success probability for recorded simulations."""
        with self.Session() as session:
            query = session.query(SimulationSummary).filter_by(token=token)
            if agent is not None:
                query = query.filter_by(agent=agent)
            sims = query.all()
            if not sims:
                return 0.0
            return float(sum(s.success_prob for s in sims) / len(sims))

    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 5) -> List[Trade]:
        if self.index is not None and self.model is not None:
            if self.index.ntotal == 0:
                return []
            vec = self.model.encode([query])[0].astype("float32")
            _distances, indices = self.index.search(np.array([vec]), k)
            ids = [int(idx) for idx in indices[0] if idx != -1]
            if not ids:
                return []
            with self.Session() as session:
                return list(session.query(Trade).filter(Trade.id.in_(ids)))
        # simple fallback search
        with self.Session() as session:
            return (
                session.query(Trade)
                .filter(Trade.context.contains(query))
                .limit(k)
                .all()
            )

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        if self._replication_sub is not None:
            self._replication_sub.__exit__(None, None, None)
